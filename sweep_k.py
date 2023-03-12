#!/usr/bin/env python3
from params_proto.hyper import Sweep
import sys
import os
from default_args import Args
this_file_name = sys.argv[0]

with Sweep(Args) as sweep:
    with sweep.product:
        Args.model = ['dgcnn']
        Args.corrupt = [['dummy'], ['noise'], ['remove_local'], ['drop_uniform'], ['drop'], ['warp']]
        Args.level = [5]
        Args.k = [1, 2, 3, 5, 10, 15, 20, 25, 30]



sweep.save(os.path.splitext(this_file_name)[0] + '.jsonl')