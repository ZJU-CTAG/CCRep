import _jsonnet
import json
import sys
import os
from copy import deepcopy
import subprocess
import torch
from allennlp.commands.train import train_model_from_file

sys.path.extend(['./'])

from utils.task_argparse import read_apca_args
from utils.file import dump_json
from base_global import py_intepreter_path

# For importing customed modules
from core import *
from utils import GlobalLogger as mylogger

args = read_apca_args()

converted_json_file_path = './temp_config.json'
serialization_dir_temp = 'models/apca/{dataset}_{model}_{subset}'
test_cmd_temp = py_intepreter_path + ' tasks/apca/apca_evaluate.py -model {model} -dataset {dataset} -subset {subset} -cuda {cuda}'
aggre_cv_results_temp = py_intepreter_path + ' tasks/apca/apca_aggre_cv_results.py -model {model} -dataset {dataset}'

test_mode = False
base_jsonnet_path = f'tasks/apca/configs/{args.model}/{args.dataset}_cv_base.jsonnet'
train_base_cv_data_path = f'data/apca/{args.dataset}/cv/'
test_dataset = args.dataset
test_model_file_names = ['model.tar.gz']

base_config_json = json.loads(_jsonnet.evaluate_file(base_jsonnet_path))

cuda_device = args.cuda
base_config_json['trainer']['cuda_device'] = cuda_device
torch.cuda.set_device(cuda_device)

for subset in ['cv_1','cv_2','cv_3','cv_4','cv_5']:
    serial_dir = serialization_dir_temp.format(dataset=args.dataset, model=args.model, subset=subset)
    ver_config_json = deepcopy(base_config_json)
    ver_config_json['train_data_path'] = os.path.join(train_base_cv_data_path, subset, 'train_patches.pkl')
    for callback in ver_config_json['trainer']['callbacks']:
        callback['serialization_dir'] = serial_dir

    if not test_mode:
        dump_json(ver_config_json, converted_json_file_path, indent=None)
        mylogger.info('apca_cv_train_helper', f'Start to train cv_{subset}')
        ret = train_model_from_file(
            converted_json_file_path,
            serial_dir,
            force=True,
            file_friendly_logging=True,
        )
        del ret
        torch.cuda.empty_cache()
        if os.path.exists(converted_json_file_path):
            os.remove(converted_json_file_path)

    for test_model_file_name in test_model_file_names:
        mylogger.info('apca_cv_train_helper', f'Start to test cv_{subset}')
        patience = 5
        while True:
            try:
                subprocess.run(
                    test_cmd_temp.format(model=args.model, dataset=args.dataset, subset=subset,
                                         cuda=str(cuda_device)),
                    shell=True, check=True
                )
                break
            except subprocess.CalledProcessError as e:
                torch.cuda.empty_cache()
                mylogger.error('apca_evaluate', f"Retry #{patience}, Error: {e}")
                patience -= 1
                if patience == 0:
                    raise e
        torch.cuda.empty_cache()

mylogger.info('apca_cv_train_helper', 'Cv training done')
# aggregating cross-validation results
subprocess.run(
    aggre_cv_results_temp.format(model=args.model, dataset=args.dataset),
    shell=True, check=True
)

sys.exit(0)
