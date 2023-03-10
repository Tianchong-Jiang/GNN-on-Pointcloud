#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss
import sklearn.metrics as metrics
from params_proto.hyper import Sweep
import wandb

from default_args import Args
from data import ModelNet40
from model import PointNet, DGCNN, DGCNN_with_TNet

arg2model = {
    'pointnet': PointNet,
    'dgcnn': DGCNN,
    'dgcnn_tnet': DGCNN_with_TNet}


def train():
    train_loader = DataLoader(ModelNet40(partition='train', operations=Args.currupt), num_workers=8,
                              batch_size=Args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', operations=Args.currupt), num_workers=8,
                             batch_size=Args.test_batch_size, shuffle=True, drop_last=False)

    #Try to load models
    model = arg2model[Args.model]().to(Args.device)

    # if Args.model == 'pointnet':
    #     model = PointNet().to(Args.device)
    # elif Args.model == 'dgcnn':
    #     model = DGCNN().to(Args.device)
    # else:
    #     raise Exception("Not implemented")

    print(f"model used:{Args.model}")

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    opt = optim.Adam(model.parameters(), lr=Args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, Args.epochs, eta_min=Args.lr)

    criterion = cal_loss

    for epoch in range(Args.epochs):
        ## train ##
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(Args.device, dtype=torch.float), label.to(Args.device, dtype=torch.float).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)

        wandb.log({'train_loss': loss, 'epoch': epoch})
        wandb.log({'accuracy': metrics.accuracy_score(train_true, train_pred), 'epoch': epoch})

        ## test ##
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(Args.device, dtype=torch.float), label.to(Args.device, dtype=torch.float).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        wandb.log({'test_accuracy': metrics.accuracy_score(train_true, train_pred), 'epoch': epoch})

        scheduler.step()

def test():
    test_loader = DataLoader(ModelNet40(partition='test', operations=Args.currupt),
                             batch_size=Args.test_batch_size, shuffle=True, drop_last=False)

    #Try to load models
    model = arg2model[Args.model]().to(Args.device)

    # if Args.model == 'pointnet':
    #     model = PointNet().to(Args.device)
    # elif Args.model == 'dgcnn':
    #     model = DGCNN().to(Args.device)
    # else:
    #     raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(Args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(Args.device), label.to(Args.device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)

    wandb.log({'test_accuracy': metrics.accuracy_score(test_true, test_pred)})


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("sweep_file",
    #                     type=str, help="sweep file")
    # parser.add_argument("-l", "--line-number",
    #                     type=int, help="line number of the sweep-file")
    # args = parser.parse_args()

    # # Obtain kwargs from Sweep
    # sweep = Sweep(Args).load(args.sweep_file)
    # kwargs = list(sweep)[args.line_number]

    # Args._update(kwargs)

    # if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    #     avail_gpus = [3]
    #     cvd = avail_gpus[args.line_number % len(avail_gpus)]
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)
    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # _init_()

    torch.manual_seed(42)

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project=f'gnn-on-pointcloud',
        group='test',
    )
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if not Args.eval:
        train()
    else:
        test()

    wandb.finish()
