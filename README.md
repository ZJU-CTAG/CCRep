
This is the replication package for **CCRep** model titled *"CCRep: Learning Code Change Representations via Pre-Trained Code Model and Query Back"* including both code and datasets. 


# Requirements

## Python Package Dependencies
You should al least install following packages to run our code:
- PyTorch: 1.8.0+cu111
- allennlp: 2.8.0
- allennlp_models: 2.8.0
- nltk: 3.5
- NumPy: 1.22.2
- jsonnet: 0.18.0
- ...

The full list of dependencies is listed in `requirements.txt`.


# Run CCRep
CCRep is evaluated on three different code-change-related tasks:
- Commit Message Generation (CMG)
- Automated Patch Correctness Assessment (APCA)
- Just-in-time Defect Prediction (JIT-DP)

## Data Preparation
Due to size limit, we archive our data in the Google Drive and you can download the data from this link: [CCRep-data.zip](https://drive.google.com/file/d/1s4k2KT3p7XrnxbDXvTvzhQexxLCk4dQd/view?usp=share_link). 
Unzip this file and move the `data` folder to the root of this project to finish preparation of data.

## Before Running
To ensure some scripts will work well, you have to do two things first:
1. Open "base_global.py" and check the path of Python interpreter. If you are not using the default "python", you should configure the right python interpreter path here.
2. **Make sure you are running all the code at the root directory of the CCRep project**, this is important. 


## APCA task
Execute follow command **at the root of the project** to run the apca task:
```shell
python tasks/apca/apca_cv_train_helper.py -model {token/line/hybrid} -dataset {Small/Large} -cuda {(your cuda device id)}
```
- model: Using which variant of CCRep: "token", "line" or "hybrid"
- dataset: Run CCRep on which dataset: "Large" or "Small" (case-sensitive!)
- cuda: ID of CUDA device you use

The script will automatically do cross-validation training, testing and report the final performance.


## CMG task
Execute following command **at the root of the project** to run the cmg task:
```shell
python tasks/cmg/cmg_train_from_config.py -dataset {corec/fira} -model {token/line/hybrid} -cuda {(your cuda device id)}
```
- dataset: Running on which dataset: "fira" or "corec"
- model: Using which variant of CCRep: "token", "line" or "hybrid"
- cuda: ID of CUDA device you use

The script will automatically do training, validation and testing, and report the final performance. 


## JIT-DP task
Execute following command **at the root of the project** to run the jit-dp task:
```shell
python tasks/jitdp/jitdp_train_from_config.py -model {token/line/hybrid} -project {(which project to run)} -cuda {(your cuda device id)}
```
- model: Using which variant of CCRep: "token", "line" or "hybrid"
- project: Run CCRep on which project: "gerrit", "jdt", "go", "platform", "openstack"
- cuda: ID of CUDA device you use

Also, this script will automatically do training, validation and testing, and report the final performance.

# Citation
If you use this repository, please consider citing our paper:

```
@inproceedings{liu2023ccrep,
  title={CCRep: Learning Code Change Representations via Pre-Trained Code Model and Query Back},
  author={Liu, Zhongxin and Tang, Zhijie and Xia, Xin and Yang, Xiaohu},
  booktitle={Proceedings of the 2023 IEEE/ACM 45th International Conference on Software Engineering},
  pages={1--13},
  year={2023}
}
```




# QA
1. - Q: My GPU memory is not enough to run the code and always encounter "CUDA out of memory" error:
   - A: 
     - If you are encountering this problem during training: Try to modify the "batch_size" in allennlp config file to solve it. 
     For example, if you are running jitdp task using "token" model on "gerrit" project, you should open 
     `tasks/jitdp/configs/token/gerrit_train.jsonnet` and decrease the `data_loader.batch_size`. 
     However, to keep the batch_size consistent with us, you should also modify `trainer.num_gradient_accumulation_steps`. The **real_batch_size = batch_size * num_gradient_accumulation_steps**, thus when you decrese `data_loader.batch_size`, 
     you should correspondingly increase `trainer.num_gradient_accumulation_steps`.
     - If you are encountering this problem during testing: Try to decrease the `batch_size` of the testing script (e.g., `cmg_predict_to_file.py`, `apca_evaluate.py` and `jitdp_evaluate.py`). 
       It uses default batch_size=32, which may be somehow large for some models.   

2. - Q: "FileNotFoundError" during running code.
   - A: First make sure you are running all the commands at the root of this project. Then try to explore the path that file located.
   
3. - Q:  Where is the data and my dumped models? 
   - A: In the `data` and `models` folders in the root of the project directory.


# Note
- For the CoReC dataset of CMG, our code reads the raw data provided by NNGen and does filtering during reading them by dataset reader, instead of directly removing filtered items from this dataset in-place.