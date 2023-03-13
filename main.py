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
import csv
from pathlib import Path


from default_args import Args
from data import ModelNet40
from model import PointNet, DGCNN, DGCNN_with_TNet

arg2model = {
    'pointnet': PointNet,
    'dgcnn': DGCNN,
    'dgcnn_tnet': DGCNN_with_TNet}

def save_data_to_csv(train_accuracy, test_accuracy):
    """This changes behavior depends on Args!!"""

    row = [f"{Args.model}", f"{'_'.join(Args.corrupt)}", f"{Args.k}", f"{Args.level}", f"{Args.kernel}", f"{train_accuracy}", f"{test_accuracy}"]

    column_heading = ["model", "corrupt", "k", "level", "kernel", "train_accuracy", "test_accuracy"]

    path = Path(f'/evaluation/{Args.exp_name}.csv')

    # Write heading if file doesn't exist
    if not path.is_file():
        with open(path, 'a+', newline='\n', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(column_heading)

    with open(path, 'a', newline='\n', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(row)


def train():
    train_set = ModelNet40(partition='train', operations=Args.corrupt)
    test_set = ModelNet40(partition='test', operations=Args.corrupt)
    train_loader = DataLoader(train_set, num_workers=0,
                              batch_size=Args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, num_workers=0,
                             batch_size=Args.test_batch_size, shuffle=True, drop_last=False)

    #Try to load models
    model = arg2model[Args.model]().to(Args.device)

    print(f"model used:{Args.model}")

    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    opt = optim.Adam(model.parameters(), lr=Args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, Args.epochs, eta_min=Args.lr)

    criterion = cal_loss

    accumulation_steps = int(16 / Args.batch_size)

    for epoch in range(Args.epochs):
        ## train ##
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        step = 0
        for data, label in train_loader:
            data = train_set.process_data(data)
            data, label = data.to(Args.device, dtype=torch.float), label.to(Args.device, dtype=torch.float).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label) / accumulation_steps
            loss.backward()
            if (step + 1) % accumulation_steps == 0:
                opt.step()
                opt.zero_grad()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
            step += 1
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_accuracy = metrics.accuracy_score(train_true, train_pred)

        wandb.log({'train_loss': loss, 'epoch': epoch})
        wandb.log({'accuracy': train_accuracy, 'epoch': epoch})

        ## test ##
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data = test_set.process_data(data)
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

        test_accuracy = metrics.accuracy_score(test_true, test_pred)
        wandb.log({'test_accuracy': test_accuracy, 'epoch': epoch})

        scheduler.step()

    save_data_to_csv(train_accuracy, test_accuracy)

def test():
    test_loader = DataLoader(ModelNet40(partition='test', operations=Args.currupt),
                             batch_size=Args.test_batch_size, shuffle=True, drop_last=False)

    #Try to load models
    model = arg2model[Args.model]().to(Args.device)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_file",
                        type=str, help="sweep file")
    parser.add_argument("-l", "--line-number",
                        type=int, help="line number of the sweep-file")
    args = parser.parse_args()

    # Obtain kwargs from Sweep
    sweep = Sweep(Args).load(args.sweep_file)
    kwargs = list(sweep)[args.line_number]

    Args._update(kwargs)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        avail_gpus = [3]
        cvd = avail_gpus[args.line_number % len(avail_gpus)]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cvd)

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project=f'gnn-on-pointcloud-T2000',
        group='test',
        config=vars(Args),
    )

    Args.exp_name = "inv_seen"

    if not Args.eval:
        train()
    else:
        test()

    wandb.finish()
