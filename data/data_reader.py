import pickle
import os
import json
from collections import defaultdict
import scipy.io as scio

__all__ = ['read_data', 'get_distributed_data_cfgs']


def _read_data_pkl(train_data_dir, test_data_dir, sub_data=None, use_secret_data=False):
    clients = []
    groups = []
    train_data_index = {}
    test_data_index = {}
    print('>>> Read data from:')

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pkl')]

    if sub_data is not None:
        taf = sub_data + '.pkl'
        assert taf in train_files
        train_files = [taf]

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data_index.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.pkl')]
    if sub_data is not None:
        taf = sub_data + '.pkl'
        assert taf in test_files
        test_files = [taf]

    if use_secret_data:
        user_eps = {}
    for f in test_files:

        file_path = os.path.join(test_data_dir, f)
        print('    ', file_path)

        with open(file_path, 'rb') as inf:
            cdata = pickle.load(inf)
        test_data_index.update(cdata['user_data'])
        if use_secret_data:
            user_eps.update(cdata['user_eps'])

    if use_secret_data:
        test_secret_data_index = {}
        train_secret_data_index = {}
        test_secret_data_dir = test_data_dir.replace('test', 'test_secret')
        train_secret_data_dir = train_data_dir.replace('train', 'train_original')

        test_secret_files = os.listdir(test_secret_data_dir)
        train_secret_files = os.listdir(train_secret_data_dir)

        test_secret_files = [f for f in test_secret_files if f.endswith('.pkl')]
        train_secret_files = [f for f in train_secret_files if f.endswith('.pkl')]

        if sub_data is not None:
            taf = sub_data + '.pkl'
            assert taf in test_secret_files
            test_secret_files = [taf]

        if sub_data is not None:
            taf = sub_data + '.pkl'
            assert taf in train_secret_files
            train_secret_files = [taf]

        for f in train_secret_files:
            file_path = os.path.join(train_secret_data_dir, f)
            print('    ', file_path)

            with open(file_path, 'rb') as inf:
                cdata = pickle.load(inf)
            train_secret_data_index.update(cdata['user_data'])

        for f in test_secret_files:
            file_path = os.path.join(test_secret_data_dir, f)
            print('    ', file_path)

            with open(file_path, 'rb') as inf:
                cdata = pickle.load(inf)
            test_secret_data_index.update(cdata['user_data'])

        return clients, groups, train_data_index, test_data_index, train_secret_data_index, test_secret_data_index, user_eps
    else:
        return clients, groups, train_data_index, test_data_index


def _read_data_mat(train_data_dir, test_data_dir, sub_data=None):
    clients = []
    groups = []
    train_data_index = {}
    test_data_index = {}
    print('>>> Read data from:')

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.pkl')]
    if sub_data is not None:
        taf = sub_data + '.mat'
        assert taf in train_files
        train_files = [taf]

    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        print('    ', file_path)

        cdata = scio.loadmat(file_path)
        # all users
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        # user_data is a dictionary
        train_data_index.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.mat')]
    if sub_data is not None:
        taf = sub_data + '.mat'
        assert taf in test_files
        test_files = [taf]

    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        print('    ', file_path)

        cdata = scio.loadmat(file_path)
        test_data_index.update(cdata['user_data'])

    clients = list(sorted(train_data_index.keys()))
    return clients, groups, train_data_index, test_data_index


def _read_dir_leaf(data_dir):
    print('>>> Read data from:', data_dir)
    clients = []
    groups = []
    # If the dict object doesn't exist, don't raise a KeyError
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def get_distributed_data_cfgs(data_name, sub_name, client_id):
    root = os.path.dirname(os.path.realpath(__file__))
    cfgs = os.path.join(root, data_name, 'data', 'distributed', sub_name)
    # all_cfgs = os.listdir(cfgs)
    return os.path.join(cfgs, client_id + '.json')


def read_data(train_data_dir, test_data_dir, data_format, sub_data=None, use_secret_data=False):
    if data_format == 'json':
        # The data here does not distinguish the corresponding format
        assert sub_data is None, 'The data in LEAF format is saved as multiple JSON files, and the subdata name cannot be specified'
        train_clients, train_groups, train_data = _read_dir_leaf(train_data_dir)
        test_clients, test_groups, test_data = _read_dir_leaf(test_data_dir)

        assert train_clients == test_clients
        assert train_groups == test_groups

        return train_clients, train_groups, train_data, test_data
    elif data_format == 'pkl':
        return _read_data_pkl(train_data_dir, test_data_dir, sub_data, use_secret_data)

    elif data_format == 'mat':
        return _read_data_mat(train_data_dir, test_data_dir, sub_data)

    else:
        raise ValueError('Only supports data formats: *.pkl, *.json', '*.mat')
