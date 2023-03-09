#!/usr/bin/env python3
from params_proto import ParamsProto
from params_proto import Proto


class Args(ParamsProto):
    model = 'dgcnn'
    model_path = 'models'
    device = 'cpu'
    eval = False

    lr = 0.001
    batch_size = 32
    test_batch_size = 16
    epochs = 250

    # model params
    dropout = 0.5
    emb_dims = 1024
    k = 20
