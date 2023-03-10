#!/usr/bin/env python3
from params_proto import ParamsProto
from params_proto import Proto


class Args(ParamsProto):
    # overall params
    model = 'dgcnn_tnet'
    model_path = 'models'
    device = 'cpu'
    eval = False

    # training params
    lr = 0.001
    batch_size = 32
    test_batch_size = 16
    epochs = 250

    # model params
    dropout = 0.5
    emb_dims = 1024
    k = 20

    # currupt params
    currupt = ['trans']
    scale = 0.01
    prob = 0.5

    # kernel params
    kernel = 'asym'

