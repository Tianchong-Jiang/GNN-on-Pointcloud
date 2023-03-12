#!/usr/bin/env python3
from params_proto.hyper import Sweep
import sys
import os
from default_args import Args
this_file_name = sys.argv[0]

with Sweep(Args) as sweep:
    with sweep.product:
        Args.model = ['pointnet', 'dgcnn', 'dgcnn_tnet']
        Args.corrupt = [['dummy'], ['trans'], ['rigid'], ['scale'], ['rigid', 'scale']]
        Args.kernel = ['global', 'local', 'dist', 'asym']

sweep.save(os.path.splitext(this_file_name)[0] + '.jsonl')