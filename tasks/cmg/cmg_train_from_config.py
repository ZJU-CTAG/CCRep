import torch
import subprocess
import _jsonnet
import json
import sys

from allennlp.commands.train import train_model_from_file

sys.path.append('./')

from utils.file import dump_json
from utils.task_argparse import read_cmg_args
from utils import GlobalLogger as mylogger
from base_global import py_intepreter_path

# For importing customed modules
from core import *


if __name__ == '__main__':
    args = read_cmg_args()

    test_cmd = f'{py_intepreter_path} tasks/cmg/cmg_predict_to_file.py -dataset {args.dataset} -model {args.model} -cuda {args.cuda}'
    jsonnet_file_path = f'tasks/cmg/configs/{args.dataset}/{args.model}_train.jsonnet'
    converted_json_file_path = f'./temp_config.json'
    serialization_dir = f'models/cmg/{args.dataset}/{args.model}/'

    config_json = json.loads(_jsonnet.evaluate_file(jsonnet_file_path))

    # Manually set cuda device to avoid additional memory usage bug on GPU:0
    # See https://github.com/pytorch/pytorch/issues/66203
    cuda_device = int(args.cuda)
    config_json['trainer']['cuda_device'] = cuda_device
    torch.cuda.set_device(cuda_device)

    mylogger.debug('train_from_config',
                   f'dataset={args.dataset}, model = {args.model}, cuda_device = {cuda_device}')

    dump_json(config_json, converted_json_file_path, indent=None)
    ret = train_model_from_file(
        converted_json_file_path,
        serialization_dir,
        force=True,
        file_friendly_logging=True,
    )
    del ret
    torch.cuda.empty_cache()

    # Do test
    subprocess.run(
        test_cmd,
        shell=True, check=True
    )

    # release resources
    sys.exit(0)