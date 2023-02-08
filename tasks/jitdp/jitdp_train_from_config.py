import subprocess
import _jsonnet
import json
import sys
import torch
from allennlp.commands.train import train_model_from_file

sys.path.extend(['./'])

from utils.task_argparse import read_jitdp_args
from utils.file import dump_json
from utils import GlobalLogger as mylogger
from base_global import py_intepreter_path

# For importing customed modules
from core import *

args = read_jitdp_args()

jsonnet_file_name = f'tasks/jitdp/configs/{args.model}/{args.project}_train.jsonnet'
converted_json_file_path = f'./temp_config.json'
serialization_dir = f'models/jitdp/{args.project}_{args.model}'

test_cmd = f'{py_intepreter_path} tasks/jitdp/jitdp_evaluate.py ' \
           f'-model {args.model} -project {args.project} -cuda {args.cuda}'

config_json = json.loads(_jsonnet.evaluate_file(jsonnet_file_name))

# add serial dir to callback parameters
for callback in config_json['trainer']['callbacks']:
    callback['serialization_dir'] = serialization_dir

cuda_device = args.cuda
config_json['trainer']['cuda_device'] = int(cuda_device)
# Manually set cuda device to avoid additional memory usage bug on GPU:0
# See https://github.com/pytorch/pytorch/issues/66203
torch.cuda.set_device(cuda_device)

mylogger.debug('train_from_config',
               f'project. = {args.project}, model = {args.model}, cuda_device = {cuda_device}')

print('start to train from file...')
dump_json(config_json, converted_json_file_path, indent=None)
ret = train_model_from_file(
    converted_json_file_path,
    serialization_dir,
    force=True,
    file_friendly_logging=True,
)
del ret
torch.cuda.empty_cache()

if os.path.exists(converted_json_file_path):
    os.remove(converted_json_file_path)

# testing
print('start to test...')
subprocess.run(
    test_cmd,
    shell=True, check=True
)

# Exit to release GPU memory
sys.exit(0)