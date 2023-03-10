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


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
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
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
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

        np.random.shuffle(pointcloud)

        for operation in self.operations:
            pointcloud = self.op_dict[operation](pointcloud)

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

    def visualize(self, num_samples = 10):
        print('visualizing pointcloud on wandb...')
        # Visualize pointcloud on Wandb
        for i in range(num_samples):
            pointcloud, label = self.__getitem__(i)
            # import pdb; pdb.set_trace()
            wandb.log({f'label:{label}': wandb.Object3D(pointcloud)}, step = i)
        print('visualizing pointcloud on wandb...done')

    def trans(self, pointcloud, scale = 1):
        trans = np.random.rand(3) * scale
        pointcloud = np.add(trans, pointcloud)

        return pointcloud

    def rigid(self, pointcloud, scale = 1):
        rot = R.random().as_matrix()
        trans = np.random.rand(3) * scale

        pointcloud = einops.rearrange(pointcloud, 'n d -> d n')
        pointcloud = rot @ pointcloud
        pointcloud = einops.rearrange(pointcloud, 'd n -> n d')

        pointcloud = np.add(trans, pointcloud)

        return pointcloud

    def noise(self, pointcloud, scale = 0.02):
        trans = np.random.randn(*pointcloud.shape) * scale
        pointcloud = pointcloud + trans
        return pointcloud

    def remove_local(self, pointcloud, scale = 0.1):
        idx = np.random.randint(pointcloud.shape[0])
        center = pointcloud[idx]
        pointcloud = pointcloud[np.linalg.norm(pointcloud - center, axis=1) > scale]
        return pointcloud

    def warp(self, pointcloud, scale = 1.1):
        pointcloud = np.power(pointcloud, scale)
        return pointcloud

    def drop_uniform(self, pointcloud, prob = 0.5):
        idx = round(prob * pointcloud.shape[0]) - 1
        pointcloud = pointcloud[:idx]
        return pointcloud

    def drop(self, pointcloud, prob = 0):
        axis = np.random.randint(0,3)
        low = np.min(pointcloud[:,axis])
        high = np.max(pointcloud[:,axis])
        selected_points = []
        for point in pointcloud:
            # draw random number base on x (or y z) position of the point
            dist_to_low = ((point[axis] - low) / (high - low))
            prob_accept = prob + (1 - prob) * dist_to_low
            if np.random.uniform() < prob_accept:
                selected_points.append(point)
        pointcloud = np.asarray(selected_points)
        return pointcloud

if __name__ == '__main__':
    dataset = ModelNet40(operation='drop')

    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project=f'gnn-on-pointcloud',
        group='test',
    )

    dataset.visualize()

    wandb.finish()
