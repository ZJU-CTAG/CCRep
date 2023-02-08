import sys
from typing import Tuple, List

from tqdm import tqdm
import torch

sys.path.append('./')

from pprint import pprint
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.models.model import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from core import *
from utils.task_argparse import read_apca_args
from utils.allennlp_utils.build_util import build_dataset_reader_from_config
from utils.file import save_evaluate_results

args = read_apca_args()

data_file_name = 'test_patches.pkl'

data_base_path = "data/apca/" + args.dataset + f"/cv/{args.subset}/"
model_base_path = f'models/apca/{args.dataset}_{args.model}_{args.subset}/'

batch_size = 32
cuda_device = args.cuda

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


model_path = model_base_path + 'model.tar.gz'
data_file_path = data_base_path + 'test_patches.pkl'

print(f'\nModel path: {model_path}')
print(f'data path: {data_file_path}\n')

dataset_reader = build_dataset_reader_from_config(
    config_path= model_base_path + 'config.json',
    serialization_dir=model_base_path
)

model = Model.from_archive(model_path)

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
    'Accuracy': accuracy_score(all_ref, all_pred),
    'Precision': precision_score(all_ref, all_pred),
    'Recall': recall_score(all_ref, all_pred),
    'F1-Score': f1_score(all_ref, all_pred),
    'AUC': roc_auc_score(all_ref, all_score)
}
print('*'*80)
pprint(result_dict)

saved_config = {
    'test_file_name': data_file_name,
    'test_model_name': model_path
}
save_evaluate_results(result_dict,
                      saved_config,
                      model_base_path+'eval_results.json')

sys.exit(0)





