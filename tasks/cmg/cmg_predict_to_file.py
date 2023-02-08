import sys
from typing import Tuple
import time

import torch
from allennlp.common import JsonDict
from allennlp.data import Vocabulary, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from tqdm import tqdm

sys.path.append('./')

from evaluation.evaluate_res import eval_cmg_corec, eval_cmg_fira
from utils.allennlp_utils.build_util import build_dataset_reader_from_config
from utils.allennlp_utils.cmg_id_token_utils import convert_str_tokens_to_line, convert_tokens_to_strs, \
    convert_prediction_tokens_to_ids, convert_prediction_ids_to_tokens
from utils import GlobalLogger as my_logger
from core import *
from utils.data_utils.lemmatization import apply_fira_lemmatization
from utils.data_utils.placeholder_utils import revert_replaceholder_as_var
from utils.task_argparse import read_cmg_args


class CMGPredictor(Predictor):
    def predict(self, code_change_block: Dict) -> JsonDict:
        return self.predict_json(
            code_change_block
        )

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        raw_data_block = {
            **json_dict
        }
        return self._dataset_reader.text_to_instance(raw_data_block)


def predict_to_file(
        model_path: str,
        config_path: str,
        vocab_path: str,
        data_paths: Tuple[str,str],
        ref_dump_path: str,
        pred_dump_path: str,
        excluded_tokens: List[str] = ['@start@', '@end@'],
        replace_token_map: Dict[str, str] = {},
        msg_namespace: str = None,
        batch_size: int = 1,
        cuda_device: int = -1,
        merge_subtoken_method: str = 'none',
        end_token: str = '@end@',
        use_fira_lemmatization: bool = False,
        revert_placeholders: bool = False,
        debug: bool = False,
        **kwargs
    ):
    my_logger.info('predict_imp_data', 'Building dataset reader...')
    dataset_reader = build_dataset_reader_from_config(config_path)
    try:
        model = Model.from_archive(model_path)
    except RuntimeError as e:
        print(f'Error: {e}\n Try to mannually feed dict to rescue...')
        vocab = Vocabulary.from_files(vocab_path)
        model = Model.from_archive(model_path, vocab=vocab)
    predictor = CMGPredictor(model, dataset_reader)

    if cuda_device != -1:
        model = model.cuda(cuda_device)
        torch.cuda.set_device(cuda_device)

    dataset_reader.debug = debug
    instances = list(dataset_reader.read(data_paths))

    ref_dump_file = open(ref_dump_path, 'w')
    pred_dump_file = open(pred_dump_path, 'w')
    batch = []
    ref_line_count = 0
    pred_line_count = 0

    batch_ref, batch_pred = [], []
    for i, instance in tqdm(enumerate(instances), total=len(instances)):
        batch.append(instance)
        msg_tokens = instance['msg'].tokens if 'msg' in instance else instance['target_tokens'].tokens
        ref = convert_tokens_to_strs(msg_tokens, excluded_tokens, replace_token_map, merge_subtoken_method, None)
        ref_id_list = convert_prediction_tokens_to_ids(ref.split(' '),
                                                       model.vocab,
                                                       namespace=msg_namespace,
                                                       excluded_tokens=excluded_tokens,
                                                       return_tensor=False)
        batch_ref.append(torch.LongTensor(ref_id_list).unsqueeze(0))

        ref = revert_replaceholder_as_var(ref, instance['meta_data']['vars'], mode='as_sentence') if revert_placeholders else ref
        ref = ref.lower()
        ref = apply_fira_lemmatization(ref) if use_fira_lemmatization else ref
        ref += '\n'
        ref_dump_file.write(ref)
        ref_line_count += 1

        if len(batch) == batch_size or i == len(instances) - 1:  # Do not forget tail data batch
            # Here is a potential bug, where batch_size=1 will cause torch.squeeze() eliminate all
            if len(batch) > 1:
                outputs = predictor.predict_batch_instance(batch)
            else:
                outputs = predictor.predict_instance(batch[0])
                outputs = [outputs]
            for j, output in enumerate(outputs):
                pred = convert_prediction_ids_to_tokens(output['predictions'][0],
                                                        _vocab=model.vocab,
                                                        namespace=msg_namespace,
                                                        excluded_tokens=[])
                pred = convert_str_tokens_to_line(pred, excluded_tokens, replace_token_map, merge_subtoken_method, end_token)
                pred = revert_replaceholder_as_var(pred, outputs[j]['meta_data']['vars'], mode='as_sentence') if revert_placeholders else pred
                pred = pred.lower()
                pred = apply_fira_lemmatization(pred) if use_fira_lemmatization else pred
                pred += '\n'
                pred_dump_file.write(pred)
                pred_line_count += 1

                batch_pred.append(torch.LongTensor(output['predictions'][0]).unsqueeze(0))

            batch.clear()
            batch_ref.clear()
            batch_pred.clear()

    print(f'Reference Line / Predicted Line = {ref_line_count} / {pred_line_count}')

    ref_dump_file.close()
    pred_dump_file.close()

    return model, dataset_reader

if __name__ == '__main__':
    args = read_cmg_args()
    batch_size = 32

    if args.dataset == 'corec':
        data_file_name = 'cleaned_test.diff'
        msg_file_name = 'cleaned_test.msg'
        merge_subtoken_method = 'none'
        end_token = '@end@'         # Also known as END_SYMBOL
        msg_namespace = "msg_tokens"
        use_fira_lemmatization = False
        revert_placeholder = False
    elif args.dataset == 'fira':
        data_file_name = 'test_diff.json'
        msg_file_name = 'test_msg.json'
        merge_subtoken_method = 'codebert'
        end_token = '</s>'
        msg_namespace = "code_tokens"
        use_fira_lemmatization = False
        revert_placeholder = True
    else:
        raise ValueError(f'No such dataset: {args.dataset}')

    model_base_path = f'models/cmg/{args.dataset}/{args.model}/'
    data_base_path = f'data/cmg/{args.dataset}/'
    ref_dump_path = f'{model_base_path}test_ref.txt'
    pred_dump_path = f'{model_base_path}test_pred.txt'

    bleu_excluded_tokens = ['@start@', '@end@', '<s>', '</s>', '<pad>', '', '@@PADDING@@', '@@UNKNOWN@@',
                            '<start>', '<eos>', '<pad>']

    model, reader = predict_to_file(
        model_path=model_base_path + 'model.tar.gz',
        config_path=model_base_path + 'config.json',
        vocab_path=model_base_path + 'vocabulary',
        data_paths=(data_base_path + data_file_name, data_base_path + msg_file_name),
        ref_dump_path=ref_dump_path,
        pred_dump_path=pred_dump_path,
        excluded_tokens=bleu_excluded_tokens,
        msg_namespace=msg_namespace,
        batch_size=batch_size,
        cuda_device=args.cuda,
        merge_subtoken_method=merge_subtoken_method,
        end_token=end_token,
        use_fira_lemmatization=use_fira_lemmatization,
        revert_placeholders=revert_placeholder,
        debug=False,
    )

    time.sleep(2)
    print('\n' + '*'*50)
    print(f'Cmg Evaluation Results (model={args.model}):')
    if args.dataset == 'corec':
        eval_cmg_corec(pred_dump_path, ref_dump_path)
    else:
        eval_cmg_fira(pred_dump_path, ref_dump_path)
