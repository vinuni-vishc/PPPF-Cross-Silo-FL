import numpy as np
import os
import time
import abc
import torch
from torch import nn, optim
import pandas as pd
from clients.base_client import BaseClient
from utils.metrics import Metrics
from utils.flops_counter import get_model_complexity_info
from utils.data_utils import MiniDataset


class BaseFedarated(abc.ABC):

    def __init__(self, options, model: nn.Module, read_dataset, append2metric=None, more_metric_to_train=None):

        self.options = options
        self.model = self.setup_model(options=options, model=model)
        self.device = options['device']
        self.clients = self.setup_clients(dataset=read_dataset, model=model)
        self.num_epochs = options['num_epochs']
        self.num_rounds = options['num_rounds']
        self.clients_per_round = options['clients_per_round']
        self.save_every_round = options['save_every']
        self.eval_on_train_every_round = options['eval_on_train_every']
        self.eval_on_test_every_round = options['eval_on_test_every']
        self.eval_on_validation_every_round = options['eval_on_validation_every']
        self.num_clients = len(self.clients)
        self.latest_model = self.clients[0].get_parameters_list()
        self.name = '_'.join(['', f'wn{options["clients_per_round"]}', f'tn{self.num_clients}'])
        self.metrics = Metrics(clients=self.clients, options=options, name=self.name, append2suffix=append2metric,
                               result_prefix=options['result_prefix'], train_metric_extend_columns=more_metric_to_train)
        self.quiet = options['quiet']

    def setup_model(self, options, model):
        dev = options['device']
        model = model.to(dev)
        input_shape = model.input_shape
        input_type = model.input_type if hasattr(model, 'input_type') else None
        self.flops, self.params_num, self.model_bytes = get_model_complexity_info(model, input_shape,
                                                                                  input_type=input_type, device=dev)
        return model

    def choose_dataset_wrapper(self):
        idx = self.options['data'].find("_")
        if idx != -1:
            dataset_name, sub_data = self.options['data'][:idx], self.options['data'][idx + 1:]
        else:
            dataset_name, sub_data = self.options['data'], None
        return DATASET_WRAPPER.get(dataset_name, MiniDataset)

    def setup_clients(self, dataset, model):
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        dataset_wrapper = self.choose_dataset_wrapper()
        all_clients = []
        for user, group in zip(users, groups):
            tr = dataset_wrapper(train_data[user], options=self.options)
            te = dataset_wrapper(test_data[user], options=self.options)
            opt = optim.SGD(self.model.parameters(), lr=self.options['lr'], momentum=0.5)
            c = BaseClient(id=user, options=self.options, train_dataset=tr, test_dataset=te, optimizer=opt, model=model,
                           model_flops=self.flops, model_bytes=self.model_bytes)
            all_clients.append(c)
        return all_clients

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        pass

    def select_clients(self, round_i, num_clients):
        num_clients = min(num_clients, self.num_clients)
        np.random.seed(round_i)
        return np.random.choice(self.clients, num_clients, replace=False).tolist()

    def aggregate_parameters_weighted(self, grads, num_samples):
        latest = []
        params_num = len(grads[0])
        m = len(grads)
        for p in range(params_num):
            new = torch.zeros_like(grads[0][p].data)
            sz = 0
            for num_sample, sol in zip(num_samples, grads):
                new += sol[p].data * num_sample
                sz += num_sample
            new /= sz
            latest.append(new)
        return latest

    def aggregate_grads_weighted(self, grads, lr, num_samples, weights_before):
        m = len(grads)
        g = []
        for i in range(len(grads[0])):
            grad_sum = torch.zeros_like(grads[0][i])
            all_sz = 0
            for ic, sz in enumerate(num_samples):
                grad_sum += grads[ic][i] * sz
                all_sz += sz
            g.append(grad_sum / all_sz)
        return [u - (v * lr) for u, v in zip(weights_before, g)]

    def aggregate_grads_simple(self, grads, lr, weights_before):
        m = len(grads)
        g = []
        for i in range(len(grads[0])):
            grad_sum = torch.zeros_like(grads[0][i])
            for ic in range(m):
                grad_sum += grads[ic][i]
            g.append(grad_sum)
        new_weights = [u - (v * lr / m) for u, v in zip(weights_before, g)]
        return new_weights

    @abc.abstractmethod
    def aggregate(self, *args, **kwargs):
        pass

    def eval_on(self, round_i, clients, use_test_data=False, use_train_data=False, use_val_data=False):
        assert use_test_data + use_train_data + use_val_data == 1
        df = pd.DataFrame(columns=['client_id', 'mean_acc', 'mean_loss', 'num_samples'])

        num_samples = []
        total_corrects = []
        losses = []
        for c in clients:
            c.set_parameters_list(self.latest_model)
            if use_test_data:
                stats = c.test(c.test_dataset_loader)
            elif use_train_data:
                stats = c.test(c.train_dataset_loader)
            elif use_val_data:
                stats = c.test(c.validation_dataset_loader)

            total_corrects.append(stats['sum_corrects'])
            num_samples.append(stats['num_samples'])
            losses.append(stats['sum_loss'])
            df = df.append({'client_id': c.id, 'mean_loss': stats['loss'], 'mean_acc': stats['acc'],
                            'num_samples': stats['num_samples'], }, ignore_index=True)
        mean_loss = sum(losses) / sum(num_samples)
        mean_acc = sum(total_corrects) / sum(num_samples)
        #
        if use_test_data:
            fn, on = 'eval_on_test_at_round_{}.csv'.format(round_i), 'test'
        elif use_train_data:
            fn, on = 'eval_on_train_at_round_{}.csv'.format(round_i), 'train'
        elif use_val_data:
            fn, on = 'eval_on_validation_at_round_{}.csv'.format(round_i), 'validation'
        #
        if not self.quiet:
            print(f'Round {round_i}, eval on "{on}" data mean loss: {mean_loss:.5f}, mean acc: {mean_acc:.3%}')
        self.metrics.update_eval_stats(round_i, df, filename=fn, on_which=on,
                                       other_to_logger={'acc': mean_acc, 'loss': mean_loss})

    def solve_epochs(self, round_i, clients, num_epochs=None):
        if num_epochs is None:
            num_epochs = self.num_epochs

        num_samples = []
        tot_corrects = []
        losses = []

        grads = []
        for c in clients:
            c.set_parameters_list(self.latest_model)
            stat, flop_stat, grad = c.solve_epochs(round_i, c.id, c.train_dataset_loader, c.optimizer, num_epochs,
                                                   hide_output=self.quiet)
            tot_corrects.append(stat['sum_corrects'])
            num_samples.append(stat['num_samples'])
            losses.append(stat['sum_loss'])
            grads.append(grad)
            self.metrics.update_cummulative_stats(round_i, flop_stat)

        mean_loss = sum(losses) / sum(num_samples)
        mean_acc = sum(tot_corrects) / sum(num_samples)

        stats = {'acc': mean_acc, 'loss': mean_loss, }
        if not self.quiet:
            print(f'Round {round_i}, train metric mean loss: {mean_loss:.5f}, mean acc: {mean_acc:.3%}')
        self.metrics.update_train_stats_only_acc_loss(round_i, stats)
        return grads, num_samples

    def test_latest_model_on_traindata(self, round_i):
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data=False)

        model_len = len(self.latest_model)
        global_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []

        for c in self.clients:
            (num, client_grad), stat = c.solve_grad()
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads += client_grad * num
        global_grads /= np.sum(np.asarray(num_samples))
        stats_from_train_data['gradnorm'] = np.linalg.norm(global_grads)

        difference = 0.
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference /= len(self.clients)
        stats_from_train_data['graddiff'] = difference
        end_time = time.time()

        self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result:
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'
                  ' Grad Norm: {:.4f} / Grad Diff: {:.4f} / Time: {:.2f}s'.format(round_i, stats_from_train_data['acc'],
                                                                                  stats_from_train_data['loss'],
                                                                                  stats_from_train_data['gradnorm'],
                                                                                  difference, end_time - begin_time))
        return global_grads

    def test_latest_model_on_evaldata(self, round_i):
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=True)
        end_time = time.time()

        if self.print_result:
            print('>>> Test on eval: round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(round_i, stats_from_eval_data['acc'],
                                                        stats_from_eval_data['loss'],
                                                        end_time - begin_time))  # print('=' * 102 + "\n")

        self.metrics.update_eval_stats(round_i, stats_from_eval_data)

    def test_latest_model_on_traindata_only_acc_loss(self, round_i):
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=False)
        end_time = time.time()

        if self.print_result:
            print('>>> Test on train: round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(round_i, stats_from_eval_data['acc'],
                                                        stats_from_eval_data['loss'],
                                                        end_time - begin_time))  # print('=' * 102 + "\n")

        self.metrics.update_train_stats_only_acc_loss(round_i, stats_from_eval_data)

    def save_model(self, round_i):
        save_path = os.path.sep.join((self.metrics.result_path, self.metrics.exp_name, f'model_at_round_{round_i}.pt'))
        # print('>>> Saving model at round: {}, saved path: {}'.format(round_i, save_path))
        torch.save(self.latest_model, save_path)
