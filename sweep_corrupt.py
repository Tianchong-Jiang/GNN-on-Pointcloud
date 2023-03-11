#!/usr/bin/env python3
from params_proto.hyper import Sweep
import sys
import os
from default_args import Args
this_file_name = sys.argv[0]

with Sweep(Args) as sweep:
    with sweep.zip:
        Args.currupt = ['trans']
        Args.scale = [0.1]


sweep.save(os.path.splitext(this_file_name)[0] + '.jsonl')