import sys
from pprint import pprint
from typing import Tuple
from tqdm import tqdm

sys.path.append('./')

from allennlp.data.data_loaders import MultiProcessDataLoader
import torch
from allennlp.models.model import Model
from sklearn.metrics import roc_auc_score

from core import *
from utils.allennlp_utils.build_util import build_dataset_reader_from_config
from utils.file import save_evaluate_results
from utils.task_argparse import read_jitdp_args

args = read_jitdp_args()

data_base_path = f'data/jitdp/{args.project}/'
model_base_path = f'models/jitdp/{args.project}_{args.model}/'
data_file_path = data_base_path + 'test.pkl'

batch_size = 32
bared_model = False

def predict_on_dataloader(model, data_loader) -> Tuple[List, List, List]:
    all_pred = []
    all_ref = []
    all_score = []
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(tqdm(data_loader)):
            outputs = model(**batch)
            all_pred.extend(outputs['pred'].cpu().detach().tolist())
            all_score.extend(outputs['probs'].cpu().detach().tolist())
            all_ref.extend(batch['label'].cpu().detach().squeeze().tolist())
    return all_ref, all_pred, all_score


dataset_reader = build_dataset_reader_from_config(
    config_path=model_base_path + 'config.json',
    serialization_dir=model_base_path
)
model = Model.from_archive(model_base_path + 'model.tar.gz')

cuda_device = args.cuda
if cuda_device != -1:
    model = model.cuda(cuda_device)
    torch.cuda.set_device(cuda_device)

data_loader = MultiProcessDataLoader(dataset_reader,
                                     data_file_path,
                                     shuffle=False,
                                     batch_size=batch_size,
                                     cuda_device=cuda_device)
data_loader.index_with(model.vocab)

if cuda_device != -1:
    model = model.cuda(cuda_device)
    torch.cuda.set_device(cuda_device)

all_ref, all_pred, all_score = predict_on_dataloader(model, data_loader)
result_dict = {
    'AUC': roc_auc_score(all_ref, all_score)
}

print('\n' + '*'*50)
print(f'JIT-DP result (model={args.model}, project={args.project}):')
pprint(result_dict)

save_evaluate_results(result_dict,
                      {
                          'test_file_name': data_file_path,
                          'test_model_name': 'model.tar.gz'
                      },
                      model_base_path+'eval_results.json')
sys.exit(0)





