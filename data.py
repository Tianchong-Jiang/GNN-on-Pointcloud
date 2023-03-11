#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from params_proto import ParamsProto
from params_proto import Proto
from scipy.spatial.transform import Rotation as R
import einops
import wandb
import matplotlib.pyplot as plt
from default_args import Args
from pathlib import Path
import torch


def download():
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_DIR = Path('/data')

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_DIR = Path('/data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, partition = 'test', operations = ['trans']):
        self.data, self.label = load_data(partition)
        self.partition = partition
        self.operations = operations

        self.op_dict = {
            'dummy': self.dummy,
            'perm': self.dummy,
            'trans': self.trans,
            'rigid': self.rigid,
            'noise': self.noise,
            'remove_local': self.remove_local,
            'warp': self.warp,
            'drop_uniform': self.drop_uniform,
            'drop': self.drop
            }

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

    def process_data(self, pointcloud):
        if not Args.corrupt == 'dummy':
            pointcloud=pointcloud[:,torch.randperm(pointcloud.shape[1]),:]

        for operation in self.operations:
            pointcloud = self.op_dict[operation](pointcloud)

        return pointcloud

    def visualize(self, num_samples = 10):
        print('visualizing pointcloud on wandb...')
        # Visualize pointcloud on Wandb
        for i in range(num_samples):
            pointcloud, label = self.__getitem__(i)
            wandb.log({f'label:{label}': wandb.Object3D(pointcloud)}, step = i)
        print('visualizing pointcloud on wandb...done')

    def render_image(self, num_samples = 5):
        print("rendering pointcloud locally...")

        pointcloud, label = self.__getitem__(2)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2])
        ax.set_axis_off()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        fig.set_size_inches(5, 5)

        fig.savefig(Path(f'/evaluation/{Args.corrupt[-1]}_{Args.param}_{2}.png'), dpi=100, pad_inches = 0)

        print("rendering pointcloud locally...done")

    def dummy(self, pointcloud):
        return pointcloud

    def trans(self, pointcloud, scale = 1):
        trans = np.random.rand(3) * scale
        pointcloud = np.add(trans, pointcloud)

        return pointcloud

    def rigid(self, pointcloud, scale = 1):
        rot = R.random().as_matrix()
        trans = np.random.rand(3) * scale

        pointcloud = einops.rearrange(pointcloud, 'b n d -> b d n')
        pointcloud = rot @ pointcloud
        pointcloud = einops.rearrange(pointcloud, 'b d n -> b n d')

        pointcloud = np.add(trans, pointcloud)

        return pointcloud

    def noise(self, pointcloud):
        levels = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3]
        trans = np.random.randn(*pointcloud.shape) * levels[Args.level]
        pointcloud = pointcloud + trans
        return pointcloud

    def remove_local(self, pointcloud):
        levels = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        idx = np.random.randint(pointcloud.shape[1])
        lengths = []
        for i in range(pointcloud.shape[0]):
            item = pointcloud[i]
            center = item[idx]
            item = item[torch.norm(item - center, dim=1) > levels[Args.level]]
            pointcloud[i,:item.shape[0]] = item
            lengths.append(item.shape[0])

        pointcloud = pointcloud[:, :min(lengths)]

        return pointcloud

    def warp(self, pointcloud):
        levels = [1, 1.05, 1.1, 1.2, 1.4, 1.8, 2.5]
        scale = np.random.uniform(1.0, levels[Args.level])
        axis = np.random.randint(0,3)
        pointcloud = pointcloud + 1
        pointcloud[:, :, axis] = torch.pow(pointcloud[:, :, axis], scale)
        pointcloud = pointcloud - 1
        return pointcloud

    def drop_uniform(self, pointcloud):
        levels = [1, 0.8, 0.5, 0.2, 0.1, 0.05, 0.01]
        prob = np.random.uniform(levels[Args.level], 1.0)
        idx = int(prob * pointcloud.shape[1]) - 1
        pointcloud = pointcloud[:, :idx]
        return pointcloud

    def drop(self, pointcloud):
        levels = [1, 0.9, 0.5, 0.3, 0.2, 0.1, 0]
        prob = np.random.uniform(levels[Args.level], 1.0)
        axis = np.random.randint(0,3)

        lengths = []
        for i in range(pointcloud.shape[0]):
            item = pointcloud[i]
            rands = torch.rand(item.shape[0])
            dist_to_low = ((item[:, axis] + 1) / 2)
            prob_accept = prob + (1 - prob) * dist_to_low
            item = item[rands < prob_accept]
            pointcloud[i,:item.shape[0]] = item
            lengths.append(item.shape[0])

        pointcloud = pointcloud[:, :min(lengths)]

        return pointcloud

if __name__ == '__main__':


    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project=f'gnn-on-pointcloud',
        group='test',
    )

    dataset = ModelNet40(operations=['noise'])
    for param in [0.01, 0.02, 0.03, 0.05]:
        Args.param = param
        Args.corrupt = ['noise']
        dataset.render_image()

    dataset = ModelNet40(operations=['remove_local'])
    for param in [0.05, 0.1, 0.2, 0.3]:
        Args.param = param
        Args.corrupt = ['remove_local']
        dataset.render_image()

    dataset = ModelNet40(operations=['drop_uniform'])
    for param in [0.8, 0.5, 0.2, 0.1]:
        Args.param = param
        Args.corrupt = ['drop_uniform']
        dataset.render_image()

    dataset = ModelNet40(operations=['drop'])
    for param in [0.8, 0.5, 0.2, 0]:
        Args.param = param
        Args.corrupt = ['drop']
        dataset.render_image()

    dataset = ModelNet40(operations=['warp'])
    for param in [1.05, 1.1, 1.2, 1.4]:
        Args.param = param
        Args.corrupt = ['warp']
        dataset.render_image()

    wandb.finish()
