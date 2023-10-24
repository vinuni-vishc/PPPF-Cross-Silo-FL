# GLOBAL PARAMETERS
import argparse

DATASETS = ['mnist', 'fmnist', 'cifar10', 'cifar100']
TRAINERS = {'fedmeta': 'FedMeta'}

OPTIMIZERS = TRAINERS.keys()
MODEL_CONFIG = {
    'mnist.cnn': {'num_classes': 10, 'image_size': 28},
    'fmnist.cnn': {'num_classes': 10, 'image_size': 28},
    'cifar10.cnn': {'num_classes': 10, 'image_size': 32},
    'cifar100.cnn': {'num_classes': 100, 'image_size': 32}
}


def base_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS, )
    parser.add_argument('--data',
                        help='name of data;',
                        type=str,
                        required=True)
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='cnn')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--device',
                        help='device',
                        default='cpu:0',
                        type=str)
    parser.add_argument('--num_rounds',
                        help='number of rounds to simulate;',
                        type=int,
                        default=200)
    parser.add_argument('--eval_on_test_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--eval_on_train_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--eval_on_validation_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--save_every',
                        help='save global model every ____ rounds;',
                        type=int,
                        default=50)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10)
    parser.add_argument('--num_epochs',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=20)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.01)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--quiet',
                        type=int,
                        default=0)
    parser.add_argument('--result_prefix',
                        type=str,
                        default='./result')
    parser.add_argument('--train_val_test',
                        action='store_true')
    parser.add_argument('--result_dir',
                        type=str,
                        default='')
    parser.add_argument('--data_format',
                        type=str,
                        default='pkl')
    parser.add_argument('--train_inner_step', default=0, type=int)
    parser.add_argument('--test_inner_step', default=0, type=int)
    parser.add_argument('--same_mini_batch', action='store_true', default=False)
    return parser


def add_dynamic_options(argparser):
    params = argparser.parse_known_args()[0]
    algo = params.algo
    if algo in ['fedmeta']:
        argparser.add_argument('--meta_algo', type=str, default='maml',
                               choices=['maml', 'reptile', 'meta_sgd'])
        argparser.add_argument('--outer_lr', type=float, required=True)
        argparser.add_argument('--meta_train_test_split', type=int, default=-1)
        argparser.add_argument('--store_to_cpu', action='store_true', default=False)
        argparser.add_argument('--use_pppfl', action='store_true', default=False)
        argparser.add_argument('--eps_smooth_factor', type=float, default=10.0)
    elif algo == 'fedavg_adv':
        argparser.add_argument('--use_all_data', action='store_true', default=False)

    return argparser
