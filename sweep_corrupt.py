#!/usr/bin/env python3
from params_proto.hyper import Sweep
import sys
import os
from default_args import Args
this_file_name = sys.argv[0]

with Sweep(Args) as sweep:
    with sweep.product:
        Args.model = ['pointnet', 'dgcnn']
        Args.corrupt = [['noise'], ['remove_local'], ['drop_uniform'], ['drop'], ['warp']]
        Args.param = [0, 1, 2, 3, 4]



sweep.save(os.path.splitext(this_file_name)[0] + '.jsonl')