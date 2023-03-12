#!/usr/bin/env python3
from params_proto import ParamsProto
from params_proto import Proto


class Args(ParamsProto):
    # experiment name
    exp_name = 'test'

    # overall params
    model = 'dgcnn'
    model_path = 'models'
    device = 'cpu'
    eval = False

    # training params
    lr = 0.001
    batch_size = 16
    test_batch_size = 2
    epochs = 5

    # model params
    dropout = 0.5
    emb_dims = 1024
    k = 20

    # currupt params
    corrupt = ['trans']
    level = 0

    # kernel params
    kernel = 'asym' # ['global', 'local', 'dist', 'asym',]

