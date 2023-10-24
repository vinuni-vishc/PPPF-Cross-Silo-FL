from trainers.fedbase import BaseFedarated, MiniDataset, optim
from clients.base_client import BaseClient
import numpy as np
import pandas as pd
import tqdm
from torch.utils.data import ConcatDataset, DataLoader


class UsingAllDataClient(BaseClient):

    def __init__(self, id, train_dataset, test_dataset, options, optimizer, model, model_flops, model_bytes):
        super(UsingAllDataClient, self).__init__(id, train_dataset, test_dataset, options, optimizer, model,
                                                 model_flops, model_bytes)
        self.all_dataset = ConcatDataset([train_dataset, test_dataset])
        self.all_dataset_loader = DataLoader(self.all_dataset, batch_size=self.num_batch_size, shuffle=False)

    def create_data_loader(self, dataset):
        return None

    def solve_epochs(self, round_i, client_id, data_loader, optimizer, num_epochs, hide_output: bool = False):
        data_loader = self.all_dataset_loader
        return super(UsingAllDataClient, self).solve_epochs(round_i, client_id, data_loader, optimizer, num_epochs,
                                                            hide_output)

    def test(self, data_loader):
        data_loader = self.all_dataset_loader
        return super(UsingAllDataClient, self).test(data_loader)


class FedAvgAdv(BaseFedarated):
    def __init__(self, options, model, read_dataset, more_metric_to_train=None):
        self.use_all_data = options['use_all_data']
        a = '[train_test_split]'
        if self.use_all_data:
            a += '_[use_all_data]'
            print('FedAvgAdv use all data for each client')
        super(FedAvgAdv, self).__init__(options=options, read_dataset=read_dataset, model=model, append2metric=a,
                                        more_metric_to_train=more_metric_to_train)
        #
        self.split_train_validation_test_clients()

    @property
    def is_train_test_split(self):
        return True

    def split_train_validation_test_clients(self, train_rate=0.8, val_rate=0.1):
        np.random.seed(self.options['seed'])
        train_rate = int(train_rate * self.num_clients)
        val_rate = int(val_rate * self.num_clients)
        test_rate = self.num_clients - train_rate - val_rate

        assert test_rate > 0 and val_rate > 0 and test_rate > 0, '不能为空'

        ind = np.random.permutation(self.num_clients)
        arr_cls = np.asarray(self.clients)
        self.train_clients = arr_cls[ind[:train_rate]].tolist()
        self.eval_clients = arr_cls[ind[train_rate:train_rate + val_rate]].tolist()
        self.test_clients = arr_cls[ind[train_rate + val_rate:]].tolist()

    def setup_clients(self, dataset, model):
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        dataset_wrapper = self.choose_dataset_wapper()
        all_clients = []
        for user, group in zip(users, groups):

            tr = dataset_wrapper(train_data[user], options=self.options)
            te = dataset_wrapper(test_data[user], options=self.options)
            opt = optim.Adam(self.model.parameters(), lr=self.options['lr'])
            if self.use_all_data:
                c = UsingAllDataClient(id=user, options=self.options, train_dataset=tr, test_dataset=te, optimizer=opt,
                                       model=model, model_flops=self.flops, model_bytes=self.model_bytes)
            else:
                c = BaseClient(id=user, options=self.options, train_dataset=tr, test_dataset=te, optimizer=opt,
                               model=model, model_flops=self.flops, model_bytes=self.model_bytes)
            all_clients.append(c)
        return all_clients

    def select_clients(self, round_i, num_clients):
        num_clients = min(num_clients, self.num_clients)
        np.random.seed(round_i)
        return np.random.choice(self.train_clients, num_clients, replace=False).tolist()

    def aggregate(self, solns, num_samples):
        return self.aggregate_parameters_weighted(solns, num_samples)

    def eval_on(self, round_i, clients, use_test_data=False, use_train_data=False, use_val_data=False):
        assert use_test_data + use_train_data + use_val_data == 1
        if not self.use_all_data:
            return super(FedAvgAdv, self).eval_on(round_i, clients, use_test_data, use_train_data, use_val_data)
        df = pd.DataFrame(columns=['client_id', 'mean_acc', 'mean_loss', 'num_samples'])

        num_samples = []
        tot_corrects = []
        losses = []
        for c in clients:
            c.set_parameters_list(self.latest_model)
            stats = c.test(c.all_dataset_loader)

            tot_corrects.append(stats['sum_corrects'])
            num_samples.append(stats['num_samples'])
            losses.append(stats['sum_loss'])
            #
            df = df.append({'client_id': c.id, 'mean_loss': stats['loss'], 'mean_acc': stats['acc'],
                            'num_samples': stats['num_samples'], }, ignore_index=True)

        mean_loss = sum(losses) / sum(num_samples)
        mean_acc = sum(tot_corrects) / sum(num_samples)
        #
        if use_test_data:
            fn, on = 'eval_on_test_at_round_{}.csv'.format(round_i), 'test'
        elif use_train_data:
            fn, on = 'eval_on_train_at_round_{}.csv'.format(round_i), 'train'
        elif use_val_data:
            fn, on = 'eval_on_validation_at_round_{}.csv'.format(round_i), 'validation'
        #
        if not self.quiet:
            print(f'Round {round_i}, eval on "{on}" dataset mean loss: {mean_loss:.5f}, mean acc: {mean_acc:.3%}')
        self.metrics.update_eval_stats(round_i, df, filename=fn, on_which=on,
                                       other_to_logger={'acc': mean_acc, 'loss': mean_loss})

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i}')

            selected_clients = self.select_clients(round_i=round_i, num_clients=self.clients_per_round)

            solns, num_samples = self.solve_epochs(round_i, clients=selected_clients)

            self.latest_model = self.aggregate(solns, num_samples)
            if (round_i + 1) % self.eval_on_test_every_round == 0:
                self.eval_on(use_test_data=True, round_i=round_i, clients=self.test_clients)

            if (round_i + 1) % self.eval_on_train_every_round == 0:
                self.eval_on(use_train_data=True, round_i=round_i, clients=self.train_clients)

            if (round_i + 1) % self.save_every_round == 0:
                # self.save_model(round_i)
                self.metrics.write()

        self.metrics.write()
