import json
import os
import pickle
from datetime import datetime
from typing import Dict

import _jsonnet

from utils import GlobalLogger as logger
from utils.os import join_path

def load_json(path):
    with open(path, 'r', encoding='UTF-8') as f:
        j = json.load(f)
    return j

def dump_json(obj, path, indent=4, sort=False):
    with open(path, 'w', encoding='UTF-8') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, sort_keys=sort)

def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def dump_text(text, dump_path):
    with open(dump_path, 'w') as f:
        f.write(text)

def read_text(path):
    with open(path, 'r') as f:
        return f.read()

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def dump(data, dump_base_path, file_name, dump_format):
    if dump_format == 'json':
        dump_file_path = join_path(dump_base_path, file_name + '.json')
        dump_json(data, dump_file_path)
    elif dump_format == 'pickle' or dump_format == 'pkl':
        dump_file_path = join_path(dump_base_path, file_name + '.pkl')
        with open(dump_file_path, 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError(f'Unsupported dump format: {dump_format}')

def read_dumped(file_path, dump_format=None):
    # guess file format when format is not given
    if dump_format is None:
        dump_format = file_path.split('.')[-1]

    if dump_format in ['pickle', 'pkl']:
        return load_pickle(file_path)
    elif dump_format == 'json':
        return load_json(file_path)
    else:
        raise ValueError(f'[read_dumped] Unsupported dump format: {dump_format}')


def jsonnet_to_json(jsonnet_path, json_dump_path):
    """
    Convert jsonnet file to json config capatible with AllenNLP's train_from_config
    method. This method will also pop the 'extra' item from jsonnet and return it along
    with json file.
    :param jsonnet_path:
    :param json_dump_path:
    :return:
    """
    config_json = json.loads(_jsonnet.evaluate_file(jsonnet_path))
    extra = config_json.pop('extra')
    dump_json(config_json, json_dump_path, indent=None)
    return config_json, extra


def save_evaluate_results(results: Dict,
                          other_configs: Dict,
                          save_json_path: str):
    results.update(other_configs)

    if os.path.exists(save_json_path):
        result_list = load_json(save_json_path)
    else:
        result_list = []

    # add time field
    results.update({
        'time': datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
    })

    result_list.append(results)
    dump_json(result_list, save_json_path)

