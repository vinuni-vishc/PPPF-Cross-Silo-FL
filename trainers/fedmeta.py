import copy
import warnings
import tqdm
import numpy as np
import pandas as pd
import torch
from clients.base_client import BaseClient
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch import optim
from torch.nn import functional as F
from trainers.fedbase import BaseFedarated
from utils.flops_counter import get_model_complexity_info
from tabulate import tabulate


class Adam:
    def __init__(self, lr=0.01, betas=(0.9, 0.999), eps=1e-08):

        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = dict()
        self.v = dict()
        self.n = 0

    def __call__(self, params, grads, i):
        if i not in self.m:
            self.m[i] = torch.zeros_like(params)
        if i not in self.v:
            self.v[i] = torch.zeros_like(params)

        self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads
        self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * torch.square(grads)

        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
        alpha = alpha / (1 - np.power(self.beta1, self.n))

        params.sub_(alpha * self.m[i] / (torch.sqrt(self.v[i]) + self.eps))

    def increase_n(self):
        self.n += 1


class Client(BaseClient):

    def __init__(self, id, train_dataset, test_dataset, options, optimizer, model, model_flops, model_bytes, **kwargs):
        super(Client, self).__init__(id, train_dataset, test_dataset, options, optimizer, model, model_flops,
                                     model_bytes)
        self.eps = kwargs.get('eps', None)
        self.train_secret_dataset = kwargs.get('train_secret_dataset', None)
        self.test_secret_dataset = kwargs.get('test_secret_dataset', None)

        self.store_to_cpu = options['store_to_cpu']
        self.weight_decay = options['wd']
        self.inner_lr = options['lr']
        self.client_optimizer = torch.optim.Adam(self.model.parameters(),
                                                 lr=self.inner_lr, weight_decay=self.weight_decay)

    def set_parameters_list(self, params_list: list):
        with torch.no_grad():
            for p, d in zip(self.model.parameters(), params_list):
                p.data.copy_(d.data)

    def get_parameters_list(self) -> list:
        with torch.no_grad():
            ps = [p.data.clone().detach() for p in self.model.parameters()]
        return ps

    def count_correct(self, preds, targets):
        _, predicted = torch.max(preds, 1)
        correct = predicted.eq(targets).sum().item()
        return correct

    def solve_meta_one_epoch(self):
        support_data_loader = self.train_dataset_loader
        query_data_loader = self.test_dataset_loader

        model, train_loss, train_acc = train_one_epoch(model=self.model, train_loader=support_data_loader,
                                                       device=self.device, criterion=self.criterion,
                                                       optimizer=self.client_optimizer, ).values()
        val_loss, val_acc, val_f1, val_auc = validate(model=model, val_loader=query_data_loader,
                                                      criterion=self.criterion, device=self.device).values()

        query_grads = compute_grads(model=model, data_loader=query_data_loader,
                                    device=self.device, criterion=self.criterion)

        comp = (len(support_data_loader.dataset) + len(query_data_loader.dataset)) * self.flops
        bytes_w = self.model_bytes
        bytes_r = self.model_bytes
        flop_stats = {'id': self.id, 'bytes_w': bytes_w, 'comp': comp, 'bytes_r': bytes_r}

        perf_stats = {'support_size': len(support_data_loader.dataset), 'support_loss': train_loss,
                      'support_acc': train_acc,
                      'query_size': len(query_data_loader.dataset), 'query_loss': val_loss, 'query_acc': val_acc,
                      'query_auc': val_auc, 'query_f1': val_f1}
        return perf_stats, flop_stats, query_grads

    def _fine_tune_and_eval_on_secret_data(self):
        assert len(self.train_secret_dataset) == len(self.train_dataset)
        assert len(self.test_secret_dataset) == len(self.test_dataset)

        train_loader = self.create_data_loader(self.train_secret_dataset)
        test_loader = self.create_data_loader(self.test_secret_dataset)

        model_clone = copy.deepcopy(self.model)
        optimizer = optim.Adam(model_clone.parameters(), lr=3e-4, weight_decay=0)

        num_epochs = 10
        results = train_fn(model=model_clone, train_loader=train_loader, val_loader=test_loader, device=self.device,
                           criterion=self.criterion, optimizer=optimizer, num_epochs=num_epochs)

        return_dict = {"support_size": len(self.train_secret_dataset),
                       "support_loss": results['train_results']['train_loss'],
                       "support_acc": results['train_results']['train_acc'],
                       "query_size": len(self.test_secret_dataset),
                       "query_loss": results['val_results']['val_loss'],
                       "query_acc": results['val_results']['val_acc'],
                       "query_f1": results['val_results']['val_f1'],
                       "query_auc": results['val_results']['val_auc']}

        return return_dict

    def _eval_on_secret_data(self):
        assert len(self.test_secret_dataset) == len(self.test_dataset)

        test_loader = self.create_data_loader(self.test_secret_dataset)
        val_results = validate(model=self.model, val_loader=test_loader, device=self.device, criterion=self.criterion)

        return {"query_loss": val_results['val_loss'],
                "query_f1": val_results['val_f1'],
                "query_acc": val_results['val_acc'],
                "query_auc": val_results['val_auc']}


class FedMeta(BaseFedarated):

    def __init__(self, options, model, read_dataset, more_metric_to_train=None, use_secret_data=True):
        self.meta_algo = options['meta_algo']
        self.outer_lr = options['outer_lr']
        self.meta_train_inner_step = options['train_inner_step']
        self.meta_test_inner_step = options['test_inner_step']
        self.meta_train_test_split = options['meta_train_test_split']
        self.store_to_cpu = options['store_to_cpu']
        self.outer_opt = Adam(lr=self.outer_lr)

        self.use_secret_data = use_secret_data
        print('>>> use_secret_data: ', self.use_secret_data)
        self.global_weight_decay = options['wd']
        self.use_pppfl = options['use_pppfl']
        self.eps_smooth_factor = options['eps_smooth_factor']

        if self.meta_train_inner_step <= 0:
            print('>>> Using FedMeta', end=' ')
            a = f'outerlr[{self.outer_lr}]_metaalgo[{self.meta_algo}]'
        else:
            print('>>> Using FedMeta, meta-train support inner step: ', self.meta_train_inner_step,
                  ', query inner step', self.meta_test_inner_step, end=' ')
            a = f'outerlr[{self.outer_lr}]_metaalgo[{self.meta_algo}]_trainstep[{self.meta_train_inner_step}]_teststep[{self.meta_test_inner_step}]'

        if options['same_mini_batch']:
            print(', use same mini-batch for inner step')
            a += '_usesameminibatch'
        else:
            print()
        super(FedMeta, self).__init__(options=options, model=model, read_dataset=read_dataset, append2metric=a,
                                      more_metric_to_train=['query_acc', 'query_loss'])
        self.split_train_validation_test_clients()
        # self.train_support, self.train_query = self.generate_batch_generator(self.train_clients)
        self.model = None

    def setup_model(self, options, model):
        dev = options['device']
        model = model.to(dev)
        input_shape = model.input_shape
        input_type = model.input_type if hasattr(model, 'input_type') else None
        self.flops, self.params_num, self.model_bytes = 0, 0, 0
        # get_model_complexity_info(model, input_shape, input_type=input_type, device=dev)
        return model

    def split_train_validation_test_clients(self):
        if self.meta_train_test_split <= 0:
            self.train_clients = self.clients
            self.test_clients = None

        else:
            np.random.seed(self.options['seed'])

            assert 0 < self.meta_train_test_split < 1
            num_train_clients = int(self.meta_train_test_split * self.num_clients)
            num_test_clients = self.num_clients - num_train_clients

            ind = np.random.permutation(self.num_clients)
            print('>>> Split clients into train, , test: ', ind[:num_train_clients], ind[-num_test_clients:])

            arr_cls = np.asarray(self.clients)
            self.train_clients = arr_cls[ind[:num_train_clients]].tolist()
            self.test_clients = arr_cls[ind[-num_test_clients:]].tolist()

    def setup_clients(self, dataset, model):
        users, groups, train_data, test_data, *args = dataset

        if self.use_secret_data:
            train_secret, test_secret, user_eps = args

        if len(groups) == 0:
            groups = [None for _ in users]
        dataset_wrapper = self.choose_dataset_wapper()
        all_clients = []
        for user, group in zip(users, groups):
            user = str(user)
            tr = dataset_wrapper(train_data[user], options=self.options, is_train=True)
            te = dataset_wrapper(test_data[user], options=self.options)

            if self.use_secret_data:
                train_secret_dataset = dataset_wrapper(train_secret[user], options=self.options, is_train=True)
                test_secret_dataset = dataset_wrapper(test_secret[user], options=self.options)

                c = Client(id=user, options=self.options, train_dataset=tr, test_dataset=te, optimizer=None,
                           model=self.model, model_flops=self.flops, model_bytes=self.model_bytes,
                           eps=user_eps[user],
                           train_secret_dataset=train_secret_dataset,
                           test_secret_dataset=test_secret_dataset,
                           weight_decay=self.global_weight_decay)

            else:
                c = Client(id=user, options=self.options, train_dataset=tr, test_dataset=te, optimizer=None,
                           model=self.model, model_flops=self.flops, model_bytes=self.model_bytes)

            all_clients.append(c)
        return all_clients

    def select_clients(self, round_i, num_clients):
        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round_i)
        return np.random.choice(self.train_clients, num_clients, replace=False).tolist()

    def test_clients_procedure(self, clients):
        df = pd.DataFrame(columns=['client_id', 'support_size', 'support_loss', 'support_acc',
                                   'query_size', 'query_loss', 'query_acc', 'query_f1', 'query_auc'])
        raise NotImplementedError

    def fine_tune_and_eval_on_secret_data(self, round_i, clients):
        df = pd.DataFrame(columns=['client_id', 'support_size', 'support_loss', 'support_acc',
                                   'query_size', 'query_loss', 'query_acc', 'query_f1', 'query_auc'])

        for c in clients:
            c.set_parameters_list(self.latest_model)
            return_dict = c._fine_tune_and_eval_on_secret_data()
            return_dict['client_id'] = c.id
            df = df.append(return_dict, ignore_index=True)

        filename = 'secret_finetune_eval_at_round_{}.csv'.format(round_i)
        df.to_csv(filename, index=False)

        # self.metrics.update_eval_stats(round_i, df=df, on_which='secret-test', filename=filename)
        if not self.quiet:
            print(df.to_string())
            print()
            print(df.mean(axis=0))

    def eval_on_secret_data(self, round_i, clients):
        df = pd.DataFrame(columns=['client_id', 'support_size', 'support_loss', 'support_acc',
                                   'query_size', 'query_loss', 'query_acc', 'query_f1', 'query_auc'])

        for c in clients:
            c.set_parameters_list(self.latest_model)
            return_dict = c._eval_on_secret_data()
            return_dict['client_id'] = c.id
            df = df.append(return_dict, ignore_index=True)

        filename = 'secret_eval_at_round_{}.csv'.format(round_i)
        df.to_csv(filename, index=False)

        if not self.quiet:
            print(df.to_string())
            print()
            print(df.mean(axis=0))

    def solve_epochs(self, round_i, clients):
        support_sizes = []
        query_grads = []
        query_sizes = []

        df = pd.DataFrame(columns=['client_id', 'support_size', 'support_loss', 'support_acc',
                                   'query_size', 'query_loss', 'query_acc', 'query_f1', 'query_auc'])

        for c in clients:
            c.set_parameters_list(self.latest_model)
            # save information
            if self.store_to_cpu:
                perf_stats, flop_stat, grads = c.solve_meta_one_epoch_save_gpu_memory()
            else:
                perf_stats, flop_stat, grads = c.solve_meta_one_epoch()

            support_sizes.append(perf_stats['support_size'])
            query_grads.append(grads)
            query_sizes.append(perf_stats['query_size'])

            self.metrics.update_cummulative_stats(round_i, flop_stat)
            perf_stats['client_id'] = c.id
            df = df.append(perf_stats, ignore_index=True)

        if not self.quiet:
            print(df.to_string())
            print()
            print(df.mean(axis=0))
            df.to_csv(f'round_{round_i}_stats.csv', index=False)

        return query_grads, query_sizes

    def aggregate_grads_simple(self, grads, lr, weights_before):
        m = len(grads)
        g = []
        for i in range(len(grads[0])):
            grad_sum = torch.zeros_like(grads[0][i])
            for ic in range(m):
                grad_sum += grads[ic][i]
            g.append(grad_sum)
        self.outer_opt.increase_n()
        for i in range(len(weights_before)):
            self.outer_opt(weights_before[i], g[i] / m, i=i)

    def aggregate_grads_weighted(self, query_grads, query_sizes, weights_before, eps):
        g = []
        if self.use_pppfl:
            warnings.warn(f'Using PPPFL, eps: {eps}, smooth_factor: {self.eps_smooth_factor}')
            eps_smoothed = temp_softmax(eps, self.eps_smooth_factor)
            query_sizes = torch.tensor(query_sizes).float().sum() * eps_smoothed
        for i in range(len(query_grads[0])):
            grad_sum = torch.zeros_like(query_grads[0][i])
            total_sz = 0
            for ic, sz in enumerate(query_sizes):
                grad = query_grads[ic][i]
                grad_sum += grad * sz
                total_sz += sz
            grad = grad_sum / total_sz
            grad = grad.add(weights_before[i].data, alpha=self.global_weight_decay)
            g.append(grad)
        self.outer_opt.increase_n()
        for i in range(len(weights_before)):
            self.outer_opt(weights_before[i], g[i], i=i)

    def aggregate(self, query_grads, query_sizes, weight_before, eps):
        self.aggregate_grads_weighted(query_grads=query_grads, query_sizes=query_sizes,
                                      weights_before=weight_before, eps=eps)

    def train(self):
        for round_i in range(self.num_rounds):
            print(f'>>> Global Training Round : {round_i}')

            selected_client_indices = self.select_clients(round_i=round_i, num_clients=self.clients_per_round)
            eps = [c.eps for c in selected_client_indices]
            eps = torch.tensor(eps)
            assert len(eps) == self.clients_per_round

            weight_before = self.latest_model
            query_grads, query_num_samples = self.solve_epochs(round_i=round_i, clients=selected_client_indices)

            self.aggregate(query_grads=query_grads, query_sizes=query_num_samples, weight_before=weight_before, eps=eps)

            if self.test_clients is not None:
                if (round_i + 1) % self.eval_on_test_every_round == 0:
                    self.test_clients_procedure(clients=self.test_clients)

            if self.use_secret_data:
                if (round_i + 1) == self.num_rounds or (round_i + 1) % 1 == 0:
                    self.fine_tune_and_eval_on_secret_data(round_i=round_i, clients=self.train_clients)
                    if self.test_clients is not None:
                        self.fine_tune_and_eval_on_secret_data(round_i=round_i, clients=self.test_clients)


def temp_softmax(input, t):
    ex = torch.exp(input / t)
    sum = torch.sum(ex, axis=0)
    return ex / sum


def train_one_epoch(model, train_loader, device, criterion=None, optimizer=None, scheduler=None):
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
    # if scheduler is None:
    #     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if scheduler is not None:
            scheduler.step(train_loss)
    train_acc = correct / total
    return {'model': model, 'train_loss': train_loss, 'train_acc': train_acc}


def validate(model, val_loader, device, criterion=None):
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
    val_acc = accuracy_score(targets.cpu(), predicted.cpu())
    val_f1 = f1_score(targets.cpu().numpy(), predicted.cpu().numpy(), average='macro')
    pred_softmax = F.softmax(outputs, dim=1).cpu().numpy()
    # val_auc = roc_auc_score(targets.cpu(), pred_softmax, average='macro', multi_class='ovo')
    val_auc = None
    return {'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1, 'val_auc': val_auc}


def train_fn(model, train_loader, val_loader=None, device=None, criterion=None, optimizer=None, scheduler=None,
             num_epochs=None):
    pbar = tqdm.tqdm(total=num_epochs)
    pbar.set_description('Training')
    for epoch in range(num_epochs):
        train_results = train_one_epoch(model=model, train_loader=train_loader, device=device, criterion=criterion,
                                        optimizer=optimizer, scheduler=scheduler)
        model = train_results['model']
        pbar.update(1)
        pbar.set_postfix(epoch=epoch, train_loss=train_results['train_loss'], trainacc=train_results['train_acc'])
    pbar.close()

    val_results = None
    if val_loader is not None:
        # validate on validation set
        val_results = validate(model, val_loader, device, criterion)
    return {'model': model, 'train_results': train_results, 'val_results': val_results}


def compute_grads(model, data_loader, device, criterion=None):
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    for batch_id, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        if criterion.reduction == 'mean':
            loss = loss * X.size(0)
        loss.backward()

    if criterion.reduction == 'mean':
        for p in model.parameters():
            p.grad /= len(data_loader.dataset)

    grads = [p.grad for p in model.parameters()]
    return grads
