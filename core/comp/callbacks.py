from typing import Dict, Any, List
import shutil
import os

from allennlp.training import TrainerCallback
from allennlp.models.archival import archive_model

from utils.stat import stat_model_param_number

@TrainerCallback.register('epoch_print')
class EpochPrintCallback(TrainerCallback):
    def __init__(self, serialization_dir=None):
        super().__init__(serialization_dir)

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        print(f'* Epoch {epoch} ended')
        print('-'*100)

@TrainerCallback.register('model_param_stat')
class ModelParamStatCallback(TrainerCallback):
    def __init__(self, serialization_dir=None):
        super().__init__(serialization_dir)

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        print('-' * 100)
        # print('Test for on_start callback...')
        stat_model_param_number(trainer.model)
        print('-' * 100)

@TrainerCallback.register('save_jsonnet_config')
class SaveJsonnetConfigCallback(TrainerCallback):
    def __init__(self,
                 file_src: str,
                 serialization_dir=None):
        super().__init__(serialization_dir)
        self._file_src = file_src
        self._serial_dir = serialization_dir

    def on_start(
        self, trainer: "GradientDescentTrainer", is_primary: bool = True, **kwargs
    ) -> None:
        # print(f'* serial_dir given to callback constructor: {self._serial_dir}')
        dst = os.path.join(self._serial_dir, '_train_config.jsonnet')
        print(f'[SaveJsonnetConfigCallback] Saving jsonnet from {self._file_src} to {dst}')
        shutil.copy(self._file_src, dst)

@TrainerCallback.register('save_epoch_model')
class SaveEpochModelCallback(TrainerCallback):
    def __init__(self,
                 serialization_dir,
                 save_epoch_points: List[int] = []):
        super().__init__(serialization_dir)
        self._serial_dir = serialization_dir
        self.save_epoch_points = save_epoch_points

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        if epoch in self.save_epoch_points:
            weight_file_name = f'epoch_{epoch}.th'
            weight_file_path = os.path.join(self.serialization_dir, weight_file_name)
            trainer._save_model_state(weight_file_path)
            archive_model(self.serialization_dir,
                          weight_file_name,
                          os.path.join(self.serialization_dir, f'model_epoch_{epoch}.tar.gz'))
            # remove the weight file after archiving
            os.remove(weight_file_path)



