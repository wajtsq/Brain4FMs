# Brain4FMs
The benchmark of self-supervised Brain Foundation Models on electrical brain signals.

## Catalogue
- [📋 Overview](#overview)
- [⚙️ Get Started](#start)
    * [🗄️ Data Preprocessing](#dataset)
    * [💻 Finetune and Evaluate](#finetune)
- [🪄 How to Extend](#extend)
    * [📚 Add new dataset](#newdata)
    * [🌟 Add new methods](#newmodel)
- [🎯 Benchmark Table](#result)

<h2 id="overview"> 📋 Overview </h2>

Brain4FMs is an open and extensible evaluation codebase for **Brain Foundation Models (BFMs)** on EEG/iEEG. The platform provides plug-and-play interfaces for data preprocessing, model loading, and standardized training/evaluation protocols, integrating 17 BFMs and 21 public datasets for reproducible comparisons. It is designed to be easily extended with new models, datasets, and protocols, and will be continuously updated to support frozen-encoder and few-shot evaluations, together with an open leaderboard as the field evolves.

<h2 id="start"> ⚙️ Get Started </h2>

Please install the following requirements, . More details can be found in `requirements.txt`.
```
torch==2.5.1
numpy==1.26.4
pandas==1.5.3
scikit-learn==1.5.2
scipy==1.13.1
```

**Quick Strart**
```
# finetune
python pretrained_run.py --run_mode finetune --model <model_name> --dataset <dataset_name> --encoder_lr <$encoder_lr> --head_lr $<head_lr>
# test
python pretrained_run.py --run_mode finetune --model <model_name> --dataset <dataset_name>
# few-shot
python pretrained_run.py --run_mode few-shot --model <model_name> --dataset <dataset_name> --shot $<shot_num>
# prototype
python pretrained_run.py --run_mode prototype --model <model_name> --dataset <dataset_name> --shot $<shot_num>
```

<h3 id="dataset"> 🗄️ Data Preprocessing </h3>

For each dataset you want to run experiments on, the first thing to do is generating a specific set of data on your device. This code provides standardized preprocessing pipelines for multiple widely-used datasets, including: [CHBMIT](https://physionet.org/content/chbmit/1.0.0/), [MAYO](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [FNUSA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [HUP](https://openneuro.org/datasets/ds004100), [SWEC](https://huggingface.co/datasets/NeuroTec/SWEC_iEEG_Dataset/tree/main), [UCSD](https://openneuro.org/datasets/ds002778/versions/1.0.5), [SleepEDF](https://physionet.org/content/sleep-edfx/1.0.0/), [ISRUC](https://sleeptight.isr.uc.pt/), [Dep-122](https://doi.org/10.18112/openneuro.ds003478.v1.1.0), [SD-28](https://doi.org/10.18150/repod.0107441), [ADHD-Adult](https://doi.org/10.17632/6k4g25fhzg.1), [ADHD-Child](https://doi.org/10.21227/rzfh-zn36), [ADFD](https://openneuro.org/datasets/ds004504/versions/1.0.2), [MDD-64](https://doi.org/10.6084/m9.figshare.4244171.v2), [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html), [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/index.html), [EEGMMIDB](https://bcmi.sjtu.edu.cn/home/seed/index.html), [EEGMat](https://physionet.org/content/eegmat/1.0.0/), [BCI-2a](https://www.bbci.de/competition/iv/#dataset1), [Cogitate-CF](https://www.arc-cogitate.com/data-bundles/#ieeg-raw-data), [Chisco](https://openneuro.org/datasets/ds005170/versions/1.1.2).

Download the datasets to obtain the raw data files for your target dataset. Then, edit `config.py` under `data_preprocess/{dataset}_preprocess` to specify:
* `data_root` - Root directory where raw datasets are stored.
* `data_save_dir` - Output directory for preprocessed data.

<h3 id="finetune"> 💻 Finetune and Evaluate </h3>

This benchmark support the training and evaluating of the models with a pretrained checkpoint. You should update the path of checkpoints in `model/model_config.py`. 

#### Pipeline

To load a checkpoint and train or evaluate from the checkpoint, please run the `pretrained_run.py`. For the `--run_mode` parameter, you can choose one from these strings:

- `finetune`: load the checkpoint, and begin finetune from the checkpoint.
- `test`: evaluate the model with this checkpoint.
- `few-shot`: freeze the encoder and finetune it with fewer samples.
- `prototype`: using prototype networks for few shot learning.

#### Note

**Fair comparison.** If you want to evaluate a result and make a direct performance comparison with other models on the same dataset, the following arguments about input data must be set according to a unified setting. These arguments includes `dataset`, `seq_len`, `patch_len`. And you can also perform learning rate search on different models by modifying `model_lr` and `clsf_lr`

**Loading checkpoints.** If you need to load ckpt (continue training from the last breakpoint), please add the `load_ckpt_path` argument (`None` if train from scratch). The path to save model checkpoints can also be set with the `save_ckpt_path` argument.

**Task MLP.** For models that do not prescribe a task-specific finetuning head, you may choose either a **Linear head** or a **CNN head** for downstream evaluation. To ensure a fair comparison across models, the benchmark reports results using the **Linear head** by default as it introduces minimal architectural bias. The architecture of the optional head can be configured with the following arguments: `cnn_in_channels`, `cnn_kernel_size` *(CNN)*.

<h2 id='extend'>🪄 How to Extend</h2>

Our benchmark is designed as an extensible framework. Users can extend the benchmark in two ways. They may **manually add** new models or datasets by following the unified configuration interfaces, or they may **use an agent-assisted workflow to semi-automatically** generate the required files. 

For agent-assisted integration, two skills are provided:

1. **Model Integration Skill**: automatically deploys a new benchmark model from its model name, GitHub repository, and checkpoint link.  
   The detailed instruction file is available at [`update_skill/models/skill.md`](update_skill/models/skill.md).

2. **Dataset Integration Skill**: automatically adds a new benchmark dataset from a downloaded raw data folder and its corresponding paper or documentation.  
   The detailed instruction file is available at [`update_skill/datasets/skill.md`](update_skill/dataset/skill.md).

Both skills generate benchmark-compatible adapters, configuration files, and validation reports, making the extension process more standardized, reproducible, and easy to verify.

---

<h3 id='newdata'>📚 Add new dataset</h3>

1. Split all the subjects in the new dataset into several groups (4-6 groups are recommended). Each group of data should be generated as a signle file named like `group_0_data.npy`, ..., `group_5_data.npy`. In each file, the shape of the numpy array is: `(seq_num, ch_num, seq_len * patch_len)` 

   The cooresponding label files should be named in similar format: `group_0_label.npy`, ..., `group_5_label.npy`. 

2. Then, add a new element in the `/utils/data_info_dict` from  `data_process/data_info.py`. Taking MAYO as an example: 

   ```python
   'MAYO': {'data_path': '.../MAYO/group_data',
        'group_num': 6,
        'split': [3, 1, 2],
        'various_ch_num': False,
        'n_class': 2,
        'sfreq': 1000,
        'channel': 1,
        'seq_len': 3,
        'downstream': 'disorder',
    },
   ```

   - `split`: how to split the `group_num` groups, as training/validation/testing respectively.
   - `various_ch_num`: whether or not the channel number may varies between different patience in this dataset.
   - `n_class`: the task performed on this dataset is a n-class classification task.
   - `sfreq`: the sampling rate of brain signals. 
   - `downstream`: the datasets correspond to specific downstream tasks, with certain models (e.g., NeuroLM) serving as prompts for guided inference.

3. To extend preprocessing to new datasets (denoted as `NAME`), create a dedicated directory `data_preprocess/NAME_preprocess/` containing: 
(1) a configuration file (`config.py`) specifying key parameters including target sampling rate (`sfreq`), notch filtering (`notch_filter`), and high-pass filtering (`high_pass_filter`);
(2) an implementation script (`preprocess.py`) that invokes the core `_segment_data()` utility function to perform signal filtering, segmentation, and sample rate normalization. This modular design ensures consistent preprocessing across datasets while permitting dataset-specific parameterization.

4. In the `utils/meta_info.py`, assume the new dataset name is `NAME`, 

   - Add a line `'NAME': default_get_data,` to the dictionary `get_data_dict`.
   - Add a line `'NAME': BinaryClassMetrics, ` (if `n_class==2`) or `'NAME': MultiClassMetrics, ` (if `n_class>=3`)  to the dictionary `metrics_dict`.


<h3 id='newmodel'>🌟 Add new methods</h3>

Assume that the method name is `NAME`,

1. Make a new directory `model/NAME/`.

2. Make a new file named `NAME.py` here, and write two classes in this file: `NAME_Trainer` and `NAME`.

   The class `NAME_Trainer` must includes these functions as members:

   - `def set_config(args: Namespace)` : A static method that sets all of the method's unique parameters as input arguments, such that any user can set these arguments. Taking LaBraM model as an example:  

   ```python
       @staticmethod
       def set_config(args: Namespace):
            args._model = 'labram_base_patch200_200'
            args.drop = 0.0
            args.drop_path = 0.1
            args.attn_drop_rate = 0.0
            args.input_size = 200
            args.model_key = 'model|module'
            args.final_dim = 1024            
            return args
   ```

   - `def clsf_loss_func(args)` : A static method that returns the loss function used by this method. Taking LaBraM model as an example:  

   ```python
       @staticmethod
       def clsf_loss_func(args):
           return nn.CrossEntropyLoss()
   ```

   - `def optimizer(args, model, clsf) ` : A static method that returns the optimizer used by this method. Taking LaBraM model as an example:

   ```python
       @staticmethod
       def optimizer(args, model, clsf):
           return torch.optim.AdamW([
               {'params': list(model.parameters()), 'lr': args.lr},
               {'params': list(clsf.parameters()), 'lr': args.clsf_lr},
           ],
               betas=(0.9, 0.95), eps=1e-5,
           )
   ```

   The class `NAME` must includes these functions as members:

   - `def forward_propagate(args, data_packet, model, clsf, loss_func=None)` : based on the data batch `data_packet` (this is determined by the `NAME_dataset` you write later), write the code for model forward propagation and loss calculation. If the code is different between the self-/unsupervision phase and fine-tuning phase, you can use the argument `args.run_mode`  to branch. Taking LaBraM model as an example:

   ```python
    @staticmethod
    def forward_propagate(args, data_packet, model, clsf, loss_func=None):
        x, y = data_packet
        emb, logit = model(x)

        if args.run_mode == 'test':
            return logit, y
        else:
            loss = loss_func(logit, y)
            return loss, logit, y
   ```

3. Then add any other files about your model in the directory `model/NAME/` to implement the method.

4. For some methods, they require unique data process (like calculating the spectral density and so on), therefore this benchmark supports to add any new Dataset class for a new method.

   Make a new file `datasets/NAME_dataset.py`, and write your dataset class `NAME_Dataset` here. Please make sure that the data tuple returned in the `__getitem__` function matches what you receive in the `forward_propagate` function. Make sure that your class contains the following basic member functions: `__len__`, `get_data_loader`. Taking `Braint1_Dataset` as an example:

   ```python
   class Brant_Dataset(Dataset):
    def __init__(self, args, x, y):
        # x: (seq_num, ch_num, seq_len, patch_len)
        # y: (seq_num, )
        self.seq_num, self.ch_num, N = x.shape
        x = _std_data_segment(x)    # time level normalization
        x = x.reshape(self.seq_num, self.ch_num, -1, args.patch_len)

        self.x = x
        self.y = y

        self.power = self.compute_power(x, fs=256)

        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)

    def __getitem__(self, index):
        return self.x    [index, :, :, :], \
               self.power[index, :, :, :], \
               self.y    [index,]

    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          shuffle=shuffle)

    @staticmethod
    def compute_power(x, fs):
        ...
        return band_sum
    ```

   If the model is trivial that it just need raw data `x` and labels `y` as the input of the `forward_propagate` function, you can directly use the class `DefaultDataset` in the `default_dataset.py`. Thus there's no need to write your own dataset class.

5. In the `utils/meta_info.py`,

   - Add a line `'NAME': NAME_Trainer,` to the dictionary `trainer_dict`, and import the class `NAME_Trainer` here.
   - Add a line `'NAME': NAME,` to the dictionary `model_dict`, and import the class `NAME` here.
   - Add a line `'NAME': NAME_Dataset,` to the dictionary `dataset_class_dict`, and import the model class `NAME_Dataset` here.

By the steps above, a new method can be added to the benchmark.

<h2 id="result"> 🎯 Benchmark Table </h2>

### Model

| Mode Name | paper | code |
| ---------- | ---------- | ---------- |
| BIOT | Biot: Biosignal transformer for cross-data learning in the wild | [BIOT](https://github.com/ycq091044/BIOT)|
| BrainBERT | Brainbert: Self-supervised representation learning for intracranial recordings | [Brainbert](https://github.com/czlwang/BrainBERT) |
| Brant | Brant: Foundation model for intracranial neural signal | [Brant](https://zju-brainnet.github.io/Brant.github.io/)
| BrainWave | BrainWave: A Brain Signal Foundation Model for Clinical Applications | - |
| Bendr | Bendr: using transformers and a contrastive self-supervised learning task to learn from massive amounts of eeg data | [Bendr](https://github.com/SPOClab-ca/BENDR) |
| LaBraM | Large brain model for learning generic representations with tremendous EEG data in BCI | [LaBraM](https://github.com/935963004/LaBraM) |
| SppEEGNet | Spp-eegnet: An input-agnostic self-supervised eeg representation model for inter-dataset transfer learning | [Spp-eegnet](https://github.com/imics-lab/eeg-transfer-learning) |
| MBrain | Mbrain: A multi-channel self-supervised learning framework for brain signals | [MBrain](https://github.com/ZJU-BrainNet/MBrain) |
| BFM | General-Purpose Brain Foundation Models for Time-Series Neuroimaging Data | [BFM](https://mohammadjavadd.github.io/old_homepage/links/bfm2024) |
| CBraMod | CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding | [CBraMod](https://github.com/wjq-learning/CBraMod) |
| Neuro-GPT | Neuro-GPT: developing/towards a foundation model for EEG | [Neuro-GPT](https://github.com/wenhui0206/NeuroGPT) |
| EEGPT | Eegpt: Pretrained transformer for universal and reliable representation of eeg signals | [Eegpt](https://github.com/BINE022/EEGPT) |
| NeuroLM | NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals | [NeuroLM](https://github.com/935963004/NeuroLM) |
| MVPFormer | A foundation model with multi-variate parallel attention to generate neuronal activity | [MVPFormer](https://github.com/IBM/multi-variate-parallel-transformer) |
| CodeBrain | CodeBrain: Bridging Decoupled Tokenizer and Multi-Scale Architecture for EEG Foundation Models | [CodeBrain](https://github.com/jingyingma01/CodeBrain) |

### Dataset
The benchmark contains 26 tasks derived from 21 public datasets.
[CHBMIT](https://physionet.org/content/chbmit/1.0.0/), [MAYO](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [FNUSA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [HUP](https://openneuro.org/datasets/ds004100), [SWEC](https://huggingface.co/datasets/NeuroTec/SWEC_iEEG_Dataset/tree/main), [UCSD](https://openneuro.org/datasets/ds002778/versions/1.0.5), [SleepEDF](https://physionet.org/content/sleep-edfx/1.0.0/), [ISRUC](https://sleeptight.isr.uc.pt/), [Dep-122](https://doi.org/10.18112/openneuro.ds003478.v1.1.0), [SD-28](https://doi.org/10.18150/repod.0107441), [ADHD-Adult](https://doi.org/10.17632/6k4g25fhzg.1), [ADHD-Child](https://doi.org/10.21227/rzfh-zn36), [ADFD](https://openneuro.org/datasets/ds004504/versions/1.0.2), [MDD-64](https://doi.org/10.6084/m9.figshare.4244171.v2), [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html), [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/index.html), [EEGMMIDB](https://bcmi.sjtu.edu.cn/home/seed/index.html), [EEGMat](https://physionet.org/content/eegmat/1.0.0/), [BCI-2a](https://www.bbci.de/competition/iv/#dataset1), [Cogitate-CF](https://www.arc-cogitate.com/data-bundles/#ieeg-raw-data), [Chisco](https://openneuro.org/datasets/ds005170/versions/1.1.2).

## Benchmark
| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| BrainWave | CHBMIT | **89.63 ± 3.08** | **79.68 ± 11.99** | **71.36 ± 9.08** | **78.35 ± 3.96** |
| REVE | CHBMIT | 83.83 ± 10.16 | 78.15 ± 12.01 | 63.05 ± 14.42 | 66.43 ± 13.25 |
| BrainBERT | CHBMIT | 78.22 ± 6.14 | 74.97 ± 10.32 | 57.59 ± 7.80 | 61.13 ± 8.79 |
| BrainOmni | CHBMIT | 75.26 ± 11.01 | 71.25 ± 11.82 | 51.22 ± 11.98 | 54.47 ± 12.02 |
| Mbrain | CHBMIT | 75.25 ± 7.73 | 70.60 ± 11.57 | 54.30 ± 10.25 | 59.74 ± 8.55 |
| LaBraM | CHBMIT | 74.54 ± 9.86 | 73.25 ± 10.50 | 53.44 ± 14.03 | 57.42 ± 17.31 |
| BFM | CHBMIT | 73.59 ± 4.78 | 72.23 ± 2.72 | 50.57 ± 9.01 | 50.56 ± 13.72 |
| EEGPT | CHBMIT | 71.18 ± 8.77 | 66.70 ± 10.91 | 45.85 ± 5.99 | 51.21 ± 14.56 |
| CBraMod | CHBMIT | 69.36 ± 4.03 | 70.33 ± 8.40 | 45.99 ± 7.69 | 48.74 ± 13.97 |
| NeuroLM | CHBMIT | 66.86 ± 6.28 | 53.52 ± 19.85 | 26.79 ± 22.54 | 38.06 ± 33.59 |
| NeuroGPT | CHBMIT | 61.00 ± 7.27 | 72.51 ± 3.11 | 18.36 ± 23.45 | 19.12 ± 25.08 |
| CodeBrain | CHBMIT | 60.35 ± 7.56 | 73.91 ± 10.28 | 51.90 ± 8.59 | 52.01 ± 10.68 |
| Brant | CHBMIT | 60.17 ± 8.94 | 71.85 ± 5.85 | 8.44 ± 11.82 | 7.42 ± 11.08 |
| BIOT | CHBMIT | 55.55 ± 8.49 | 28.57 ± 5.29 | 41.39 ± 2.64 | 61.44 ± 2.12 |
| MVPFormer | CHBMIT | 54.69 ± 1.89 | 70.22 ± 2.21 | 28.54 ± 3.24 | 25.69 ± 4.46 |
| Bendr | CHBMIT | 54.51 ± 3.12 | 59.46 ± 3.49 | 32.07 ± 1.73 | 35.55 ± 3.69 |
| SppEEGNet | CHBMIT | 43.48 ± 4.87 | 60.48 ± 5.90 | 31.21 ± 3.71 | 33.12 ± 5.21 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| BrainWave | FNUSA | **92.46 ± 4.66** | 88.68 ± 6.28 | **82.69 ± 6.13** | **82.97 ± 3.97** |
| Mbrain | FNUSA | 91.00 ± 8.23 | 86.72 ± 8.07 | 75.20 ± 12.91 | 75.51 ± 16.22 |
| EEGPT | FNUSA | 90.45 ± 4.49 | 84.35 ± 5.97 | 74.63 ± 10.75 | 77.85 ± 7.86 |
| BrainBERT | FNUSA | 90.18 ± 3.81 | **89.04 ± 4.27** | 77.46 ± 8.73 | 79.24 ± 8.11 |
| LaBraM | FNUSA | 89.27 ± 7.54 | 81.66 ± 9.79 | 71.86 ± 14.62 | 75.93 ± 11.34 |
| BrainOmni | FNUSA | 88.16 ± 6.59 | 83.59 ± 5.34 | 72.55 ± 8.33 | 74.59 ± 8.51 |
| Bendr | FNUSA | 88.05 ± 6.11 | 83.61 ± 5.70 | 73.26 ± 11.44 | 76.76 ± 9.99 |
| REVE | FNUSA | 87.08 ± 7.73 | 82.15 ± 6.76 | 71.19 ± 10.10 | 73.81 ± 8.46 |
| BIOT | FNUSA | 87.05 ± 6.69 | 82.90 ± 8.42 | 73.38 ± 13.99 | 77.34 ± 9.83 |
| Brant | FNUSA | 86.95 ± 12.35 | 83.73 ± 6.78 | 74.71 ± 6.32 | 78.89 ± 8.28 |
| CodeBrain | FNUSA | 82.84 ± 9.81 | 84.88 ± 6.47 | 78.17 ± 5.97 | 76.48 ± 9.09 |
| SppEEGNet | FNUSA | 81.50 ± 9.06 | 83.58 ± 9.19 | 73.18 ± 9.21 | 72.93 ± 9.29 |
| BFM | FNUSA | 77.79 ± 12.49 | 69.22 ± 11.08 | 62.03 ± 5.26 | 69.22 ± 8.25 |
| NeuroGPT | FNUSA | 71.91 ± 12.37 | 76.54 ± 12.12 | 29.71 ± 37.60 | 26.95 ± 34.89 |
| CBraMod | FNUSA | 71.20 ± 13.14 | 77.79 ± 7.85 | 62.46 ± 13.02 | 63.53 ± 11.62 |
| NeuroLM | FNUSA | 64.05 ± 14.38 | 71.52 ± 16.80 | 37.35 ± 27.71 | 37.82 ± 35.05 |
| MVPFormer | FNUSA | 59.47 ± 6.92 | 62.98 ± 6.38 | 36.98 ± 6.61 | 38.45 ± 11.39 |

| Model Name | Dataset | AUROC | Acc | F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| REVE | ISRUC | **91.55 ± 1.22** | **69.02 ± 2.96** | 67.95 ± 3.30 | **59.78 ± 3.59** |
| BrainOmni | ISRUC | 91.18 ± 1.88 | 67.64 ± 2.68 | 65.78 ± 4.18 | 58.09 ± 2.97 |
| BIOT | ISRUC | 91.03 ± 1.84 | 67.92 ± 3.71 | 66.76 ± 4.03 | 58.50 ± 4.89 |
| CodeBrain | ISRUC | 89.76 ± 2.04 | 65.20 ± 3.16 | 64.88 ± 5.07 | 55.73 ± 4.46 |
| BrainWave | ISRUC | 88.55 ± 0.87 | 62.33 ± 5.47 | 60.63 ± 3.53 | 52.24 ± 6.37 |
| EEGPT | ISRUC | 87.31 ± 2.16 | 60.04 ± 4.96 | 56.12 ± 8.26 | 48.61 ± 7.03 |
| Mbrain | ISRUC | 86.78 ± 4.02 | 63.06 ± 6.65 | 58.67 ± 7.55 | 51.43 ± 8.41 |
| CBraMod | ISRUC | 85.68 ± 4.54 | 54.17 ± 8.24 | 53.31 ± 7.39 | 41.87 ± 9.37 |
| LaBraM | ISRUC | 85.13 ± 2.32 | 61.24 ± 2.09 | 58.67 ± 4.10 | 50.25 ± 2.78 |
| Brant | ISRUC | 83.81 ± 1.76 | 54.78 ± 1.18 | 51.66 ± 3.94 | 42.70 ± 1.63 |
| Bendr | ISRUC | 80.53 ± 2.30 | 52.08 ± 4.21 | **73.26 ± 11.44** | 26.87 ± 7.62 |
| BrainBERT | ISRUC | 80.37 ± 2.56 | 46.81 ± 5.11 | 48.72 ± 4.51 | 33.21 ± 5.81 |
| BFM | ISRUC | 77.60 ± 2.83 | 47.49 ± 4.11 | 45.80 ± 2.73 | 26.41 ± 14.81 |
| MVPFormer | ISRUC | 70.49 ± 2.18 | 37.77 ± 4.35 | 36.90 ± 3.14 | 20.84 ± 4.37 |
| NeuroLM | ISRUC | 66.92 ± 4.76 | 27.27 ± 11.11 | 18.29 ± 8.41 | 7.35 ± 7.35 |
| SppEEGNet | ISRUC | 63.45 ± 3.20 | 26.09 ± 5.29 | 29.18 ± 2.65 | 9.77 ± 3.79 |
| NeuroGPT | ISRUC | 60.75 ± 7.33 | 30.70 ± 7.46 | 18.16 ± 8.98 | 5.98 ± 10.23 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| BrainWave | MAYO | **97.72 ± 0.99** | 93.04 ± 2.29 | 80.97 ± 8.79 | **86.32 ± 3.37** |
| BrainBERT | MAYO | 96.97 ± 0.97 | **94.97 ± 0.86** | **81.17 ± 6.86** | 82.54 ± 6.53 |
| LaBraM | MAYO | 95.99 ± 1.83 | 92.98 ± 2.41 | 75.61 ± 11.99 | 80.36 ± 8.70 |
| Bendr | MAYO | 92.50 ± 3.18 | 90.49 ± 2.38 | 68.82 ± 12.60 | 75.34 ± 11.21 |
| Mbrain | MAYO | 92.32 ± 4.11 | 91.64 ± 1.90 | 70.45 ± 10.69 | 73.39 ± 8.72 |
| REVE | MAYO | 92.08 ± 3.75 | 89.42 ± 3.56 | 66.17 ± 13.22 | 71.92 ± 11.09 |
| Brant | MAYO | 92.08 ± 3.36 | 81.84 ± 12.31 | 58.48 ± 18.68 | 69.02 ± 13.70 |
| EEGPT | MAYO | 91.76 ± 3.16 | 90.16 ± 2.64 | 66.91 ± 10.34 | 70.04 ± 6.82 |
| BrainOmni | MAYO | 91.40 ± 6.06 | 90.80 ± 2.20 | 66.93 ± 12.71 | 68.95 ± 12.07 |
| BIOT | MAYO | 89.65 ± 7.20 | 88.49 ± 5.05 | 62.95 ± 14.38 | 72.39 ± 13.37 |
| CBraMod | MAYO | 89.14 ± 3.73 | 87.20 ± 1.85 | 58.76 ± 12.13 | 63.80 ± 10.13 |
| CodeBrain | MAYO | 88.68 ± 2.16 | 90.86 ± 3.37 | 73.82 ± 11.15 | 76.20 ± 10.83 |
| BFM | MAYO | 81.28 ± 7.99 | 68.35 ± 7.73 | 42.56 ± 14.69 | 58.78 ± 13.92 |
| SppEEGNet | MAYO | 79.82 ± 3.11 | 85.53 ± 4.68 | 56.65 ± 14.08 | 60.24 ± 10.14 |
| NeuroGPT | MAYO | 77.14 ± 14.89 | 87.82 ± 3.19 | 23.12 ± 26.31 | 18.52 ± 21.13 |
| MVPFormer | MAYO | 67.54 ± 8.19 | 71.90 ± 5.83 | 33.59 ± 7.38 | 38.61 ± 9.62 |
| NeuroLM | MAYO | 63.75 ± 9.25 | 72.21 ± 15.86 | 30.18 ± 20.44 | 37.23 ± 25.89 |

| Model Name | Dataset | AUROC | Acc | F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| BIOT | SleepEDF | **94.95 ± 1.06** | 77.67 ± 3.17 | **75.03 ± 2.04** | 69.93 ± 4.26 |
| Mbrain | SleepEDF | 94.81 ± 1.06 | **78.50 ± 3.86** | 74.19 ± 2.82 | **70.89 ± 5.90** |
| CodeBrain | SleepEDF | 94.19 ± 1.74 | 73.26 ± 7.45 | 72.36 ± 4.11 | 64.62 ± 8.65 |
| Brant | SleepEDF | 93.77 ± 1.47 | 72.65 ± 7.23 | 71.89 ± 3.92 | 63.86 ± 8.34 |
| REVE | SleepEDF | 93.58 ± 1.55 | 75.22 ± 3.55 | 71.75 ± 4.09 | 66.44 ± 4.76 |
| CBraMod | SleepEDF | 93.41 ± 1.42 | 73.44 ± 4.23 | 70.06 ± 3.37 | 64.22 ± 5.37 |
| LaBraM | SleepEDF | 92.80 ± 1.04 | 75.69 ± 3.87 | 69.75 ± 3.13 | 66.92 ± 4.97 |
| BrainBERT | SleepEDF | 91.38 ± 2.09 | 68.21 ± 7.18 | 66.60 ± 4.08 | 57.83 ± 7.89 |
| Bendr | SleepEDF | 90.93 ± 1.34 | 69.31 ± 5.64 | 65.48 ± 2.57 | 59.09 ± 6.25 |
| EEGPT | SleepEDF | 90.92 ± 0.37 | 68.10 ± 2.79 | 66.82 ± 0.72 | 57.83 ± 2.59 |
| BrainWave | SleepEDF | 89.14 ± 1.87 | 66.78 ± 5.43 | 58.53 ± 5.92 | 51.56 ± 10.36 |
| BFM | SleepEDF | 88.52 ± 1.50 | 64.54 ± 3.72 | 60.42 ± 2.20 | 55.77 ± 5.27 |
| BrainOmni | SleepEDF | 87.43 ± 0.42 | 62.70 ± 3.12 | 61.80 ± 0.30 | 50.63 ± 2.49 |
| MVPFormer | SleepEDF | 80.56 ± 2.46 | 48.95 ± 7.63 | 49.39 ± 3.67 | 33.86 ± 6.84 |
| SppEEGNet | SleepEDF | 76.03 ± 2.65 | 42.76 ± 11.25 | 44.80 ± 5.52 | 28.27 ± 7.55 |
| NeuroGPT | SleepEDF | 64.60 ± 3.26 | 48.05 ± 3.56 | 27.09 ± 3.00 | 9.95 ± 6.81 |
| NeuroLM | SleepEDF | 60.57 ± 5.65 | 17.60 ± 3.78 | 15.24 ± 6.79 | 18.90 ± 26.27 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| BrainOmni | UCSD-OFF | **63.57 ± 8.57** | 57.37 ± 6.38 | 54.35 ± 15.76 | 56.84 ± 26.36 |
| CBraMod | UCSD-OFF | 62.65 ± 5.36 | 57.73 ± 4.05 | 45.62 ± 8.78 | 40.13 ± 11.56 |
| REVE | UCSD-OFF | 62.49 ± 19.12 | **58.23 ± 17.19** | 58.00 ± 18.89 | 59.96 ± 24.16 |
| EEGPT | UCSD-OFF | 60.74 ± 12.55 | 55.29 ± 7.61 | 55.86 ± 10.18 | 57.90 ± 17.09 |
| LaBraM | UCSD-OFF | 59.14 ± 5.09 | 56.49 ± 3.36 | 57.29 ± 8.08 | 59.72 ± 15.06 |
| BrainWave | UCSD-OFF | 57.98 ± 5.97 | 53.03 ± 9.54 | **61.67 ± 12.08** | 70.80 ± 18.97 |
| BFM | UCSD-OFF | 53.33 ± 3.38 | 48.81 ± 12.09 | 35.87 ± 9.19 | 46.51 ± 17.11 |
| SppEEGNet | UCSD-OFF | 53.12 ± 6.80 | 50.58 ± 5.85 | 59.49 ± 14.54 | **71.24 ± 25.21** |
| Bendr | UCSD-OFF | 52.46 ± 4.10 | 50.91 ± 2.75 | 48.03 ± 4.59 | 46.86 ± 4.91 |
| Brant | UCSD-OFF | 51.79 ± 9.73 | 44.67 ± 21.92 | 42.33 ± 28.89 | 57.52 ± 34.82 |
| BrainBERT | UCSD-OFF | 51.23 ± 6.25 | 54.50 ± 5.67 | 49.36 ± 16.71 | 49.32 ± 20.33 |
| CodeBrain | UCSD-OFF | 50.98 ± 8.38 | 54.36 ± 6.62 | 54.55 ± 14.14 | 56.53 ± 21.08 |
| NeuroLM | UCSD-OFF | 50.65 ± 6.91 | 50.15 ± 4.99 | 53.12 ± 15.12 | 60.97 ± 22.89 |
| BIOT | UCSD-OFF | 50.52 ± 18.51 | 49.48 ± 4.92 | 55.87 ± 26.97 | 68.51 ± 34.83 |
| Mbrain | UCSD-OFF | 50.40 ± 4.10 | 43.52 ± 3.69 | 31.68 ± 23.15 | 33.82 ± 28.96 |
| MVPFormer | UCSD-OFF | 49.07 ± 4.49 | 52.84 ± 6.84 | 53.33 ± 8.79 | 54.50 ± 10.14 |
| NeuroGPT | UCSD-OFF | 44.78 ± 7.23 | 50.64 ± 4.60 | 0.58 ± 1.30 | 0.37 ± 0.82 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| BrainWave | UCSD-ON | **68.71 ± 10.42** | 52.60 ± 7.68 | 60.48 ± 6.06 | 68.79 ± 14.18 |
| REVE | UCSD-ON | 62.81 ± 18.92 | 55.87 ± 13.27 | 44.78 ± 28.58 | 45.61 ± 33.20 |
| CodeBrain | UCSD-ON | 62.33 ± 10.53 | 56.25 ± 10.37 | 59.91 ± 11.19 | 63.29 ± 13.52 |
| CBraMod | UCSD-ON | 60.04 ± 11.73 | 48.98 ± 8.00 | 18.11 ± 8.27 | 13.48 ± 6.77 |
| MVPFormer | UCSD-ON | 57.51 ± 7.43 | **56.98 ± 4.94** | 50.30 ± 5.26 | 45.99 ± 4.68 |
| Brant | UCSD-ON | 55.77 ± 11.46 | 51.94 ± 10.63 | 28.79 ± 27.51 | 31.10 ± 33.85 |
| BrainOmni | UCSD-ON | 53.19 ± 14.34 | 49.34 ± 11.92 | 49.03 ± 16.01 | 50.86 ± 20.32 |
| EEGPT | UCSD-ON | 51.12 ± 21.42 | 48.47 ± 17.17 | 49.21 ± 14.90 | 49.40 ± 13.73 |
| Bendr | UCSD-ON | 51.12 ± 4.74 | 48.64 ± 4.70 | 43.15 ± 4.96 | 41.08 ± 6.35 |
| LaBraM | UCSD-ON | 50.67 ± 21.83 | 45.18 ± 10.52 | 32.57 ± 19.27 | 31.49 ± 19.10 |
| BrainBERT | UCSD-ON | 50.61 ± 4.84 | 50.57 ± 6.29 | 51.39 ± 11.05 | 53.78 ± 16.73 |
| SppEEGNet | UCSD-ON | 48.84 ± 8.07 | 49.92 ± 10.58 | 45.56 ± 22.87 | 48.39 ± 27.06 |
| Mbrain | UCSD-ON | 48.83 ± 15.48 | 46.37 ± 13.64 | 36.57 ± 17.26 | 34.19 ± 17.78 |
| NeuroLM | UCSD-ON | 47.62 ± 12.69 | 50.59 ± 17.60 | 46.84 ± 24.42 | 48.30 ± 28.51 |
| BIOT | UCSD-ON | 46.12 ± 18.33 | 48.77 ± 4.54 | **64.55 ± 4.85** | **81.37 ± 3.90** |
| NeuroGPT | UCSD-ON | 45.97 ± 9.87 | 46.78 ± 5.61 | 28.80 ± 24.98 | 30.60 ± 28.21 |
| BFM | UCSD-ON | 43.66 ± 8.29 | 48.45 ± 6.83 | 42.21 ± 17.00 | 41.72 ± 19.84 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| BrainWave | Dep-BDI | **72.40 ± 3.48** | 66.03 ± 3.03 | **61.80 ± 2.87** | 67.47 ± 5.82 |
| BrainOmni | Dep-BDI | 67.58 ± 12.08 | 62.70 ± 8.95 | 54.94 ± 9.86 | 58.32 ± 13.18 |
| REVE | Dep-BDI | 67.07 ± 12.18 | 64.58 ± 7.59 | 47.14 ± 14.04 | 44.91 ± 16.75 |
| LaBraM | Dep-BDI | 64.73 ± 10.38 | 58.98 ± 10.86 | 59.91 ± 9.04 | **70.65 ± 8.26** |
| BrainBERT | Dep-BDI | 63.70 ± 8.81 | 61.28 ± 5.16 | 47.77 ± 11.05 | 47.91 ± 12.24 |
| EEGPT | Dep-BDI | 62.57 ± 4.25 | 61.68 ± 5.26 | 50.03 ± 5.10 | 51.01 ± 10.94 |
| BFM | Dep-BDI | 62.32 ± 10.19 | 58.79 ± 6.39 | 48.40 ± 10.28 | 50.73 ± 12.92 |
| BIOT | Dep-BDI | 59.90 ± 10.70 | 29.76 ± 17.22 | 11.04 ± 2.01 | 21.86 ± 4.56 |
| Mbrain | Dep-BDI | 59.36 ± 6.63 | 62.44 ± 2.82 | 24.48 ± 16.42 | 20.45 ± 15.75 |
| NeuroLM | Dep-BDI | 54.36 ± 4.74 | 61.06 ± 2.32 | 29.95 ± 25.82 | 30.76 ± 31.38 |
| Bendr | Dep-BDI | 54.18 ± 3.12 | 54.47 ± 2.65 | 44.92 ± 2.87 | 47.47 ± 3.06 |
| CBraMod | Dep-BDI | 53.57 ± 5.15 | **68.96 ± 14.03** | 12.59 ± 7.85 | 16.05 ± 9.63 |
| Brant | Dep-BDI | 52.56 ± 8.72 | 62.39 ± 1.21 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| CodeBrain | Dep-BDI | 51.56 ± 10.60 | 59.84 ± 4.07 | 40.45 ± 12.88 | 41.00 ± 15.20 |
| NeuroGPT | Dep-BDI | 51.42 ± 8.06 | 62.39 ± 1.21 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| MVPFormer | Dep-BDI | 51.20 ± 1.53 | 55.07 ± 1.74 | 38.68 ± 6.48 | 38.44 ± 9.24 |
| SppEEGNet | Dep-BDI | 49.61 ± 2.42 | 48.69 ± 3.83 | 41.44 ± 3.60 | 45.27 ± 3.89 |

| Model Name | Dataset | AUROC | Acc | F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| BrainWave | Dep-STAI | **63.55 ± 6.15** | 51.34 ± 4.83 | **43.19 ± 3.58** | **14.97 ± 5.32** |
| LaBraM | Dep-STAI | 60.61 ± 8.63 | 36.32 ± 10.35 | 39.44 ± 4.62 | 8.08 ± 9.25 |
| BrainBERT | Dep-STAI | 60.47 ± 5.70 | 47.52 ± 11.04 | 36.72 ± 8.81 | 6.41 ± 12.62 |
| BrainOmni | Dep-STAI | 59.63 ± 5.74 | 52.77 ± 5.06 | 37.65 ± 4.63 | 10.51 ± 6.16 |
| BFM | Dep-STAI | 57.63 ± 7.73 | 45.30 ± 8.61 | 38.79 ± 5.77 | 9.85 ± 11.66 |
| NeuroGPT | Dep-STAI | 56.59 ± 5.54 | **60.52 ± 0.96** | 25.13 ± 0.25 | 0.00 ± 0.00 |
| EEGPT | Dep-STAI | 56.47 ± 5.68 | 51.40 ± 6.37 | 36.14 ± 5.89 | 2.47 ± 8.03 |
| BIOT | Dep-STAI | 56.34 ± 9.04 | 36.91 ± 7.44 | 38.61 ± 6.88 | 7.31 ± 8.81 |
| Mbrain | Dep-STAI | 55.39 ± 3.76 | 60.34 ± 2.61 | 31.65 ± 6.08 | 5.37 ± 5.93 |
| REVE | Dep-STAI | 54.33 ± 11.41 | 54.42 ± 10.31 | 35.13 ± 8.12 | 7.86 ± 20.02 |
| CBraMod | Dep-STAI | 54.15 ± 3.69 | 60.38 ± 1.66 | 35.42 ± 2.88 | 4.92 ± 5.24 |
| Bendr | Dep-STAI | 53.78 ± 3.28 | 47.73 ± 3.28 | 36.40 ± 3.39 | 4.71 ± 4.14 |
| MVPFormer | Dep-STAI | 53.68 ± 2.89 | 50.15 ± 3.76 | 34.51 ± 1.48 | 2.82 ± 3.93 |
| Brant | Dep-STAI | 53.49 ± 2.97 | **60.52 ± 0.96** | 25.13 ± 0.25 | 0.00 ± 0.00 |
| CodeBrain | Dep-STAI | 52.08 ± 5.18 | 47.50 ± 8.36 | 33.94 ± 3.94 | 3.38 ± 5.55 |
| SppEEGNet | Dep-STAI | 51.45 ± 2.35 | 46.38 ± 2.85 | 33.80 ± 1.49 | 0.72 ± 2.43 |
| NeuroLM | Dep-STAI | 50.32 ± 4.22 | 57.65 ± 4.45 | 26.81 ± 3.51 | 0.85 ± 1.81 |

| Model Name | Dataset | AUROC | Acc | F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| REVE | EEGMMIDB-R | **82.39 ± 0.50** | **58.60 ± 1.07** | **58.95 ± 0.90** | **44.80 ± 1.42** |
| BrainOmni | EEGMMIDB-R | 74.32 ± 0.99 | 49.36 ± 1.00 | 49.69 ± 1.72 | 32.47 ± 1.35 |
| CBraMod | EEGMMIDB-R | 57.50 ± 1.11 | 29.54 ± 1.66 | 30.75 ± 1.17 | 6.06 ± 2.20 |
| CodeBrain | EEGMMIDB-R | 56.60 ± 1.09 | 30.07 ± 1.95 | 30.07 ± 1.52 | 6.70 ± 2.56 |
| Bendr | EEGMMIDB-R | 55.15 ± 0.58 | 29.48 ± 1.01 | 29.42 ± 1.02 | 5.95 ± 1.35 |
| LaBraM | EEGMMIDB-R | 54.67 ± 1.85 | 28.09 ± 1.91 | 27.98 ± 2.47 | 4.12 ± 2.70 |
| MVPFormer | EEGMMIDB-R | 54.36 ± 1.29 | 28.54 ± 0.97 | 28.62 ± 1.02 | 4.74 ± 1.30 |
| BFM | EEGMMIDB-R | 52.50 ± 0.69 | 26.82 ± 0.56 | 26.81 ± 0.61 | 2.42 ± 0.75 |
| SppEEGNet | EEGMMIDB-R | 51.75 ± 0.42 | 26.07 ± 0.56 | 25.78 ± 0.75 | 1.39 ± 0.76 |
| Mbrain | EEGMMIDB-R | 51.58 ± 1.01 | 25.60 ± 1.20 | 23.44 ± 5.51 | 0.82 ± 1.57 |
| NeuroLM | EEGMMIDB-R | 51.52 ± 2.16 | 25.61 ± 1.37 | 21.63 ± 7.19 | 0.90 ± 1.73 |
| EEGPT | EEGMMIDB-R | 51.11 ± 0.76 | 25.47 ± 1.30 | 25.06 ± 2.76 | 0.62 ± 1.73 |
| NeuroGPT | EEGMMIDB-R | 50.46 ± 0.12 | 25.18 ± 0.20 | 23.53 ± 1.98 | 0.28 ± 0.31 |
| Brant | EEGMMIDB-R | 50.33 ± 0.83 | 25.05 ± 0.15 | 10.03 ± 0.04 | 0.00 ± 0.00 |
| BIOT | EEGMMIDB-R | 50.13 ± 0.83 | 25.19 ± 0.82 | 25.19 ± 1.21 | 0.27 ± 1.11 |
| BrainWave | EEGMMIDB-R | 50.00 ± 0.55 | 24.82 ± 0.86 | 25.74 ± 1.53 | -0.36 ± 1.08 |
| BrainBERT | EEGMMIDB-R | 50.00 ± 0.54 | 25.12 ± 0.43 | 24.59 ± 1.14 | 0.07 ± 0.56 |

| Model Name | Dataset | AUROC | Acc | F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| REVE | EEGMMIDB-I | **82.10 ± 1.50** | **58.83 ± 2.30** | **59.53 ± 1.91** | **45.11 ± 3.06** |
| BrainOmni | EEGMMIDB-I | 75.65 ± 1.17 | 51.34 ± 1.41 | 51.69 ± 1.28 | 35.12 ± 1.89 |
| LaBraM | EEGMMIDB-I | 59.17 ± 2.00 | 31.16 ± 1.81 | 30.78 ± 1.29 | 7.83 ± 3.18 |
| CBraMod | EEGMMIDB-I | 58.41 ± 1.67 | 30.47 ± 2.07 | 31.72 ± 1.80 | 7.34 ± 2.78 |
| CodeBrain | EEGMMIDB-I | 57.87 ± 2.02 | 30.08 ± 1.50 | 30.50 ± 1.87 | 6.85 ± 1.96 |
| MVPFormer | EEGMMIDB-I | 55.63 ± 2.40 | 29.46 ± 2.20 | 29.53 ± 2.10 | 5.95 ± 2.92 |
| Bendr | EEGMMIDB-I | 55.02 ± 0.65 | 29.04 ± 0.73 | 28.97 ± 0.78 | 5.39 ± 0.99 |
| BFM | EEGMMIDB-I | 52.45 ± 1.35 | 26.72 ± 0.70 | 26.63 ± 0.70 | 2.29 ± 0.93 |
| EEGPT | EEGMMIDB-I | 51.85 ± 0.75 | 26.69 ± 1.60 | 27.42 ± 0.89 | 2.33 ± 2.23 |
| Mbrain | EEGMMIDB-I | 51.07 ± 1.04 | 25.65 ± 0.94 | 23.15 ± 2.73 | 0.94 ± 1.11 |
| NeuroLM | EEGMMIDB-I | 51.03 ± 1.13 | 25.30 ± 0.63 | 19.60 ± 5.88 | 0.50 ± 0.81 |
| BrainWave | EEGMMIDB-I | 50.99 ± 0.51 | 25.18 ± 1.05 | 25.13 ± 2.51 | 0.09 ± 1.38 |
| Brant | EEGMMIDB-I | 50.93 ± 0.68 | 24.92 ± 0.37 | 10.02 ± 0.07 | 0.00 ± 0.00 |
| SppEEGNet | EEGMMIDB-I | 50.91 ± 0.99 | 25.67 ± 1.01 | 25.46 ± 1.19 | 0.81 ± 1.30 |
| BIOT | EEGMMIDB-I | 50.05 ± 0.59 | 24.64 ± 0.31 | 24.70 ± 0.58 | -0.55 ± 0.44 |
| BrainBERT | EEGMMIDB-I | 49.19 ± 0.44 | 24.91 ± 0.35 | 23.71 ± 1.10 | -0.25 ± 0.47 |
| NeuroGPT | EEGMMIDB-I | 49.12 ± 1.39 | 24.57 ± 0.67 | 20.59 ± 5.16 | -0.68 ± 0.76 |

| Model Name | Dataset | AUROC | Acc | F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| REVE | SEED-IV | **53.95 ± 1.03** | **28.09 ± 1.53** | **28.03 ± 1.56** | **3.93 ± 1.91** |
| CodeBrain | SEED-IV | 53.03 ± 1.65 | 26.49 ± 0.64 | 27.09 ± 1.31 | 2.03 ± 1.39 |
| BrainOmni | SEED-IV | 52.93 ± 1.49 | 27.71 ± 1.52 | 27.69 ± 1.43 | 3.47 ± 1.85 |
| BrainWave | SEED-IV | 52.73 ± 2.33 | 27.44 ± 1.77 | 27.24 ± 1.71 | 2.26 ± 2.63 |
| Mbrain | SEED-IV | 52.14 ± 1.46 | 26.49 ± 1.49 | 25.46 ± 2.11 | 1.38 ± 1.97 |
| BFM | SEED-IV | 51.78 ± 1.06 | 26.62 ± 0.70 | 25.93 ± 0.22 | 1.21 ± 0.38 |
| CBraMod | SEED-IV | 51.13 ± 0.98 | 26.13 ± 0.94 | 25.56 ± 1.45 | 1.01 ± 1.01 |
| BrainBERT | SEED-IV | 51.05 ± 2.66 | 24.98 ± 1.59 | 24.47 ± 0.89 | 0.28 ± 1.19 |
| EEGPT | SEED-IV | 50.72 ± 1.21 | 25.16 ± 1.03 | 25.42 ± 1.19 | 0.58 ± 1.61 |
| LaBraM | SEED-IV | 50.67 ± 1.06 | 25.30 ± 1.41 | 25.60 ± 0.95 | 0.77 ± 1.39 |
| MVPFormer | SEED-IV | 50.51 ± 0.13 | 25.36 ± 0.52 | 25.35 ± 0.42 | 0.47 ± 0.58 |
| NeuroGPT | SEED-IV | 50.32 ± 0.19 | 26.63 ± 0.01 | 19.51 ± 8.04 | 0.01 ± 0.02 |
| NeuroLM | SEED-IV | 50.25 ± 0.50 | 25.06 ± 0.49 | 18.27 ± 4.89 | 0.13 ± 0.17 |
| Brant | SEED-IV | 50.24 ± 1.11 | 25.58 ± 0.93 | 10.18 ± 0.29 | 0.00 ± 0.00 |
| Bendr | SEED-IV | 50.03 ± 0.45 | 24.90 ± 0.39 | 24.96 ± 0.40 | -0.06 ± 0.55 |
| SppEEGNet | SEED-IV | 49.87 ± 0.22 | 24.63 ± 0.25 | 24.90 ± 0.14 | -0.09 ± 0.21 |
| BIOT | SEED-IV | 48.34 ± 0.38 | 23.84 ± 0.86 | 23.62 ± 0.37 | -1.83 ± 0.32 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| BrainWave | SD-28 | **88.16 ± 7.45** | **81.38 ± 6.10** | **83.21 ± 8.42** | 81.13 ± 11.12 |
| REVE | SD-28 | 81.01 ± 12.95 | 75.39 ± 12.76 | 79.78 ± 9.78 | **85.69 ± 11.01** |
| NeuroGPT | SD-28 | 80.49 ± 16.78 | 76.52 ± 13.49 | 79.85 ± 10.90 | 83.84 ± 13.56 |
| BrainBERT | SD-28 | 72.84 ± 11.26 | 65.99 ± 15.43 | 62.55 ± 25.88 | 61.59 ± 28.72 |
| BrainOmni | SD-28 | 69.33 ± 14.96 | 69.28 ± 12.09 | 72.54 ± 13.62 | 76.60 ± 18.78 |
| BFM | SD-28 | 69.01 ± 14.87 | 62.29 ± 4.73 | 72.95 ± 3.72 | 80.97 ± 7.70 |
| CodeBrain | SD-28 | 64.91 ± 19.75 | 55.97 ± 8.22 | 61.46 ± 6.18 | 61.68 ± 14.68 |
| Mbrain | SD-28 | 57.93 ± 9.76 | 58.33 ± 10.29 | 64.19 ± 12.59 | 66.29 ± 15.05 |
| EEGPT | SD-28 | 56.15 ± 21.40 | 55.75 ± 10.99 | 62.02 ± 11.74 | 66.06 ± 15.41 |
| BIOT | SD-28 | 56.10 ± 8.70 | 61.62 ± 12.39 | 67.54 ± 18.41 | 65.14 ± 20.56 |
| CBraMod | SD-28 | 54.36 ± 13.85 | 58.34 ± 4.62 | 57.87 ± 17.84 | 57.73 ± 23.89 |
| LaBraM | SD-28 | 54.12 ± 7.16 | 54.46 ± 3.65 | 59.28 ± 6.78 | 58.98 ± 12.64 |
| Brant | SD-28 | 50.53 ± 11.47 | 46.05 ± 8.49 | 46.67 ± 17.48 | 50.19 ± 29.78 |
| Bendr | SD-28 | 49.49 ± 4.53 | 49.82 ± 3.40 | 48.24 ± 4.64 | 45.26 ± 4.01 |
| SppEEGNet | SD-28 | 48.81 ± 2.98 | 47.12 ± 3.54 | 32.75 ± 6.19 | 26.81 ± 4.89 |
| NeuroLM | SD-28 | 48.61 ± 4.39 | 49.44 ± 4.58 | 52.88 ± 28.48 | 62.40 ± 35.05 |
| MVPFormer | SD-28 | 48.18 ± 4.76 | 50.59 ± 4.46 | 53.47 ± 9.26 | 53.95 ± 12.57 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| EEGPT | ADHD-Adult | **96.48 ± 3.07** | 91.36 ± 3.89 | 90.34 ± 4.88 | 88.85 ± 8.18 |
| BrainOmni | ADHD-Adult | 96.16 ± 2.39 | 91.12 ± 2.38 | 90.31 ± 2.61 | 89.23 ± 4.07 |
| BrainWave | ADHD-Adult | 96.02 ± 2.88 | 91.02 ± 4.02 | 90.59 ± 4.10 | 91.39 ± 4.95 |
| BIOT | ADHD-Adult | 95.74 ± 2.11 | 89.65 ± 4.33 | 89.14 ± 4.45 | 89.69 ± 4.83 |
| REVE | ADHD-Adult | 95.70 ± 1.97 | 91.46 ± 2.33 | 90.80 ± 2.47 | 90.36 ± 4.77 |
| Mbrain | ADHD-Adult | 95.49 ± 3.17 | 91.58 ± 3.97 | 90.96 ± 4.33 | 90.88 ± 5.92 |
| Brant | ADHD-Adult | 95.15 ± 2.67 | 89.10 ± 3.88 | 88.62 ± 4.04 | 89.53 ± 4.67 |
| LaBraM | ADHD-Adult | 94.95 ± 3.59 | 90.74 ± 3.84 | **92.21 ± 3.89** | 91.41 ± 3.65 |
| CBraMod | ADHD-Adult | 94.65 ± 2.44 | 91.89 ± 3.64 | 91.21 ± 4.35 | 91.03 ± 6.89 |
| CodeBrain | ADHD-Adult | 88.50 ± 8.38 | **92.38 ± 3.21** | 91.85 ± 3.41 | **91.77 ± 3.84** |
| BFM | ADHD-Adult | 87.90 ± 4.47 | 81.54 ± 3.81 | 80.35 ± 4.11 | 80.52 ± 4.53 |
| NeuroGPT | ADHD-Adult | 87.54 ± 5.10 | 81.96 ± 4.69 | 82.08 ± 3.94 | 85.10 ± 3.01 |
| NeuroLM | ADHD-Adult | 81.72 ± 13.64 | 74.58 ± 6.76 | 74.42 ± 5.85 | 76.80 ± 10.58 |
| BrainBERT | ADHD-Adult | 74.37 ± 4.60 | 68.83 ± 5.20 | 61.01 ± 11.08 | 56.61 ± 13.64 |
| Bendr | ADHD-Adult | 62.14 ± 3.49 | 61.33 ± 4.30 | 59.78 ± 4.94 | 60.84 ± 6.46 |
| SppEEGNet | ADHD-Adult | 51.32 ± 4.02 | 54.20 ± 5.98 | 35.94 ± 29.70 | 36.25 ± 35.42 |
| MVPFormer | ADHD-Adult | 50.12 ± 1.31 | 51.13 ± 4.62 | 51.06 ± 2.80 | 52.94 ± 2.48 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| REVE | ADHD-Child | **79.19 ± 6.94** | **71.36 ± 4.29** | 73.81 ± 5.73 | 73.51 ± 8.21 |
| BrainWave | ADHD-Child | 78.27 ± 4.45 | 70.72 ± 3.59 | 73.73 ± 6.24 | 74.49 ± 11.98 |
| BIOT | ADHD-Child | 74.54 ± 6.98 | 66.93 ± 4.12 | **75.92 ± 3.01** | **85.61 ± 4.21** |
| BrainOmni | ADHD-Child | 70.83 ± 8.81 | 61.65 ± 6.83 | 62.70 ± 8.98 | 60.71 ± 14.46 |
| BFM | ADHD-Child | 69.49 ± 11.06 | 68.73 ± 8.73 | 74.95 ± 7.31 | 75.28 ± 4.12 |
| Mbrain | ADHD-Child | 69.02 ± 10.65 | 62.69 ± 7.84 | 66.45 ± 5.50 | 65.83 ± 3.79 |
| Brant | ADHD-Child | 68.69 ± 9.87 | 58.39 ± 8.85 | 51.08 ± 13.62 | 44.27 ± 15.15 |
| EEGPT | ADHD-Child | 67.37 ± 13.27 | 63.65 ± 10.41 | 67.59 ± 10.29 | 68.16 ± 11.87 |
| LaBraM | ADHD-Child | 65.92 ± 12.04 | 64.46 ± 7.85 | 73.52 ± 6.01 | 70.05 ± 6.73 |
| CodeBrain | ADHD-Child | 64.16 ± 8.53 | 62.88 ± 5.61 | 69.38 ± 7.24 | 73.77 ± 11.06 |
| NeuroLM | ADHD-Child | 63.68 ± 5.85 | 60.18 ± 4.66 | 70.89 ± 3.67 | 79.79 ± 4.83 |
| CBraMod | ADHD-Child | 63.55 ± 5.05 | 63.84 ± 5.65 | 72.89 ± 2.89 | 80.65 ± 5.03 |
| BrainBERT | ADHD-Child | 55.02 ± 3.15 | 55.79 ± 3.22 | 56.08 ± 7.40 | 53.28 ± 9.45 |
| MVPFormer | ADHD-Child | 53.61 ± 2.24 | 56.83 ± 2.95 | 60.51 ± 2.06 | 59.78 ± 2.25 |
| SppEEGNet | ADHD-Child | 52.92 ± 5.04 | 48.90 ± 3.70 | 47.90 ± 6.89 | 44.60 ± 8.05 |
| Bendr | ADHD-Child | 52.85 ± 1.55 | 50.98 ± 1.28 | 51.65 ± 2.26 | 48.77 ± 2.39 |
| NeuroGPT | ADHD-Child | 47.42 ± 11.66 | 55.68 ± 6.01 | 63.74 ± 10.73 | 69.26 ± 18.07 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| REVE | ADFD | **84.28 ± 7.12** | **77.36 ± 7.89** | **79.11 ± 6.58** | 78.70 ± 8.94 |
| LaBraM | ADFD | 83.07 ± 9.64 | 75.33 ± 7.43 | 77.00 ± 7.54 | 76.48 ± 9.29 |
| EEGPT | ADFD | 81.11 ± 5.08 | 71.89 ± 4.86 | 72.56 ± 5.06 | 69.87 ± 7.32 |
| BrainOmni | ADFD | 79.90 ± 7.40 | 71.38 ± 4.43 | 72.91 ± 4.98 | 71.90 ± 9.42 |
| BrainWave | ADFD | 74.11 ± 4.67 | 67.90 ± 4.59 | 68.10 ± 9.93 | 73.26 ± 9.84 |
| BrainBERT | ADFD | 59.31 ± 4.70 | 58.03 ± 4.59 | 60.34 ± 11.47 | 64.04 ± 14.65 |
| BFM | ADFD | 58.66 ± 8.18 | 58.13 ± 9.02 | 59.45 ± 11.51 | 61.14 ± 8.87 |
| BIOT | ADFD | 55.72 ± 17.42 | 48.54 ± 11.76 | 57.58 ± 13.67 | 69.74 ± 23.55 |
| CBraMod | ADFD | 54.78 ± 5.64 | 58.69 ± 3.76 | 40.98 ± 20.49 | 36.82 ± 19.93 |
| Mbrain | ADFD | 53.46 ± 12.08 | 50.12 ± 10.27 | 52.54 ± 12.43 | 51.96 ± 13.98 |
| Bendr | ADFD | 52.08 ± 2.12 | 52.32 ± 0.93 | 53.64 ± 4.53 | 51.98 ± 6.98 |
| MVPFormer | ADFD | 51.96 ± 1.38 | 52.59 ± 2.96 | 55.67 ± 4.50 | 55.12 ± 6.08 |
| SppEEGNet | ADFD | 51.68 ± 1.72 | 50.62 ± 1.85 | 51.16 ± 5.19 | 49.09 ± 7.11 |
| NeuroLM | ADFD | 50.88 ± 3.97 | 52.77 ± 3.44 | 63.41 ± 8.66 | 71.36 ± 15.92 |
| Brant | ADFD | 50.45 ± 6.96 | 53.55 ± 3.67 | 69.69 ± 3.21 | 84.23 ± 4.00 |
| CodeBrain | ADFD | 50.23 ± 8.75 | 68.71 ± 5.35 | 71.81 ± 3.83 | 72.26 ± 4.31 |
| NeuroGPT | ADFD | 50.04 ± 3.34 | 54.68 ± 1.24 | 70.70 ± 1.04 | **85.77 ± 0.61** |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| REVE | MDD-64 | **96.88 ± 4.33** | **88.49 ± 8.50** | **90.13 ± 6.47** | 92.42 ± 4.10 |
| LaBraM | MDD-64 | 95.95 ± 4.21 | 86.99 ± 7.77 | 88.19 ± 6.69 | 89.40 ± 8.40 |
| EEGPT | MDD-64 | 94.38 ± 5.39 | 86.91 ± 7.61 | 88.39 ± 6.24 | 90.08 ± 4.44 |
| BrainOmni | MDD-64 | 93.89 ± 6.98 | 84.97 ± 9.05 | 87.00 ± 7.39 | 89.37 ± 4.76 |
| BrainWave | MDD-64 | 93.52 ± 2.65 | 85.21 ± 1.75 | 83.54 ± 3.06 | 80.53 ± 5.66 |
| BIOT | MDD-64 | 93.33 ± 3.01 | 81.23 ± 7.14 | 85.15 ± 4.90 | **93.11 ± 2.33** |
| Mbrain | MDD-64 | 93.20 ± 7.79 | 86.92 ± 10.60 | 88.94 ± 7.62 | 91.21 ± 4.99 |
| BFM | MDD-64 | 92.56 ± 7.80 | 86.22 ± 10.63 | 88.37 ± 7.58 | 90.93 ± 4.78 |
| BrainBERT | MDD-64 | 91.98 ± 7.05 | 84.71 ± 7.10 | 85.38 ± 6.40 | 84.84 ± 10.13 |
| Brant | MDD-64 | 91.57 ± 9.10 | 79.69 ± 10.35 | 79.70 ± 13.60 | 79.96 ± 19.39 |
| CBraMod | MDD-64 | 89.71 ± 12.75 | 84.11 ± 11.79 | 85.76 ± 9.24 | 86.54 ± 8.71 |
| Bendr | MDD-64 | 84.82 ± 6.80 | 77.82 ± 4.67 | 79.95 ± 2.36 | 81.25 ± 5.08 |
| NeuroLM | MDD-64 | 82.39 ± 10.61 | 80.08 ± 6.25 | 80.42 ± 9.06 | 81.15 ± 17.53 |
| CodeBrain | MDD-64 | 77.39 ± 15.28 | 84.53 ± 8.92 | 86.09 ± 6.74 | 87.23 ± 8.51 |
| MVPFormer | MDD-64 | 72.96 ± 9.29 | 82.71 ± 5.53 | 84.30 ± 4.22 | 85.60 ± 6.06 |
| NeuroGPT | MDD-64 | 69.90 ± 12.45 | 67.03 ± 13.47 | 76.92 ± 7.42 | 89.05 ± 3.91 |
| SppEEGNet | MDD-64 | 57.51 ± 1.58 | 60.44 ± 9.49 | 57.53 ± 11.62 | 53.53 ± 12.49 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| REVE | EEGMat | **77.95 ± 9.10** | **75.41 ± 7.90** | **57.31 ± 10.30** | **66.00 ± 10.06** |
| LaBraM | EEGMat | 70.51 ± 3.12 | 67.71 ± 4.12 | 47.82 ± 7.14 | 52.83 ± 12.87 |
| Mbrain | EEGMat | 68.45 ± 6.03 | 74.90 ± 2.44 | 12.68 ± 14.64 | 9.63 ± 11.19 |
| BrainOmni | EEGMat | 68.22 ± 7.40 | 59.55 ± 6.57 | 47.51 ± 3.87 | 59.59 ± 6.03 |
| EEGPT | EEGMat | 67.48 ± 3.61 | 63.64 ± 6.38 | 49.15 ± 2.00 | 59.16 ± 3.60 |
| BrainWave | EEGMat | 66.33 ± 6.38 | 51.48 ± 20.10 | 43.61 ± 6.91 | 54.67 ± 8.48 |
| CBraMod | EEGMat | 63.47 ± 6.36 | 72.11 ± 4.05 | 19.31 ± 18.02 | 16.27 ± 16.92 |
| NeuroLM | EEGMat | 63.23 ± 8.05 | 31.20 ± 10.30 | 42.14 ± 4.49 | 63.60 ± 4.31 |
| BrainBERT | EEGMat | 61.42 ± 7.32 | 66.73 ± 6.27 | 44.56 ± 8.65 | 49.56 ± 12.95 |
| BFM | EEGMat | 59.98 ± 2.71 | 65.79 ± 4.27 | 34.96 ± 5.56 | 36.22 ± 9.89 |
| BIOT | EEGMat | 59.41 ± 7.70 | 26.69 ± 0.59 | 42.13 ± 0.73 | 64.54 ± 0.68 |
| CodeBrain | EEGMat | 58.62 ± 5.37 | 62.00 ± 9.20 | 48.82 ± 3.58 | 58.04 ± 6.75 |
| Brant | EEGMat | 57.13 ± 12.45 | 74.67 ± 0.59 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| MVPFormer | EEGMat | 55.62 ± 2.97 | 55.73 ± 3.44 | 37.45 ± 3.85 | 43.46 ± 4.07 |
| Bendr | EEGMat | 53.69 ± 1.95 | 65.83 ± 4.95 | 34.13 ± 6.75 | 34.47 ± 6.33 |
| SppEEGNet | EEGMat | 52.32 ± 3.48 | 49.70 ± 7.33 | 36.94 ± 4.12 | 47.43 ± 8.97 |
| NeuroGPT | EEGMat | 51.04 ± 5.91 | 73.14 ± 0.64 | 0.00 ± 0.00 | 0.00 ± 0.00 |

| Model Name | Dataset | AUROC | Acc | F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| BrainBERT | DEAP | **52.02 ± 2.12** | 22.97 ± 2.24 | 26.70 ± 2.37 | 1.73 ± 2.01 |
| CodeBrain | DEAP | 51.54 ± 2.88 | 26.51 ± 11.79 | 19.57 ± 4.76 | 0.39 ± 2.75 |
| SppEEGNet | DEAP | 51.35 ± 2.85 | 26.78 ± 5.76 | 25.80 ± 2.11 | 0.72 ± 2.69 |
| NeuroGPT | DEAP | 51.34 ± 3.14 | **43.59 ± 3.62** | 15.16 ± 0.90 | 0.00 ± 0.00 |
| BrainWave | DEAP | 51.26 ± 2.66 | 27.11 ± 6.76 | 26.10 ± 5.49 | 1.05 ± 7.71 |
| CBraMod | DEAP | 51.04 ± 0.64 | 18.76 ± 1.88 | 24.58 ± 5.09 | -0.14 ± 0.87 |
| Mbrain | DEAP | 50.75 ± 4.59 | 41.79 ± 4.16 | 14.71 ± 1.03 | 0.00 ± 0.00 |
| NeuroLM | DEAP | 50.70 ± 3.42 | 33.34 ± 13.49 | 15.26 ± 3.62 | -0.26 ± 0.70 |
| LaBraM | DEAP | 50.57 ± 2.03 | 28.77 ± 4.48 | 25.74 ± 2.97 | 1.61 ± 1.42 |
| BIOT | DEAP | 50.29 ± 3.34 | 30.06 ± 5.76 | 22.96 ± 4.50 | -1.39 ± 2.98 |
| MVPFormer | DEAP | 50.25 ± 1.59 | 25.99 ± 5.07 | 25.92 ± 1.62 | 1.10 ± 1.71 |
| REVE | DEAP | 50.24 ± 5.00 | 26.17 ± 5.38 | 24.37 ± 5.03 | -1.26 ± 3.48 |
| Bendr | DEAP | 50.18 ± 1.84 | 26.35 ± 2.09 | 25.61 ± 2.44 | -0.37 ± 3.49 |
| EEGPT | DEAP | 49.83 ± 1.76 | 26.47 ± 5.33 | **26.70 ± 4.89** | **2.89 ± 2.70** |
| BFM | DEAP | 49.57 ± 2.04 | 28.20 ± 6.66 | 24.83 ± 1.10 | 0.49 ± 1.46 |
| Brant | DEAP | 48.41 ± 2.27 | 34.42 ± 17.45 | 12.26 ± 5.56 | 0.00 ± 0.00 |
| BrainOmni | DEAP | 45.12 ± 3.29 | 22.33 ± 2.66 | 22.34 ± 3.37 | -4.59 ± 2.22 |

| Model Name | Dataset | AUROC | Acc | F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| REVE | BCI-2a | **66.42 ± 4.63** | **33.67 ± 2.09** | **38.13 ± 2.37** | **11.46 ± 1.16** |
| EEGPT | BCI-2a | 56.73 ± 3.50 | 27.69 ± 1.76 | 26.69 ± 2.89 | 3.62 ± 2.41 |
| LaBraM | BCI-2a | 56.12 ± 3.85 | 28.55 ± 1.58 | 28.40 ± 1.85 | 4.74 ± 2.10 |
| CodeBrain | BCI-2a | 54.53 ± 2.00 | 27.41 ± 0.93 | 26.13 ± 2.94 | 3.21 ± 1.24 |
| BrainWave | BCI-2a | 53.98 ± 2.34 | 27.34 ± 1.66 | 25.23 ± 6.27 | 3.12 ± 2.22 |
| Mbrain | BCI-2a | 53.85 ± 1.77 | 27.04 ± 1.71 | 21.02 ± 5.24 | 2.72 ± 2.28 |
| BrainOmni | BCI-2a | 53.80 ± 1.11 | 26.59 ± 2.18 | 26.97 ± 2.37 | 2.12 ± 2.91 |
| CBraMod | BCI-2a | 53.09 ± 1.47 | 26.79 ± 1.49 | 27.81 ± 1.84 | 2.39 ± 1.98 |
| Brant | BCI-2a | 52.58 ± 1.40 | 25.00 ± 0.00 | 10.00 ± 0.00 | 0.00 ± 0.00 |
| NeuroLM | BCI-2a | 51.87 ± 1.75 | 27.23 ± 1.32 | 24.58 ± 6.26 | 2.97 ± 1.76 |
| MVPFormer | BCI-2a | 51.82 ± 1.18 | 26.56 ± 0.84 | 26.30 ± 0.65 | 2.08 ± 1.11 |
| SppEEGNet | BCI-2a | 51.28 ± 1.37 | 25.81 ± 1.42 | 25.53 ± 1.19 | **11.46 ± 1.16** |
| BFM | BCI-2a | 51.24 ± 0.61 | 25.98 ± 0.87 | 25.91 ± 0.77 | 1.31 ± 1.15 |
| NeuroGPT | BCI-2a | 51.20 ± 2.83 | 25.55 ± 1.42 | 25.50 ± 4.91 | 0.73 ± 1.89 |
| BIOT | BCI-2a | 51.00 ± 0.88 | 25.67 ± 0.90 | 25.41 ± 0.74 | 0.90 ± 1.20 |
| Bendr | BCI-2a | 50.85 ± 0.58 | 25.96 ± 0.98 | 25.93 ± 0.94 | 1.28 ± 1.31 |
| BrainBERT | BCI-2a | 49.86 ± 1.14 | 25.43 ± 1.09 | 25.26 ± 0.88 | 0.24 ± 1.03 |

| Model Name | Dataset | AUROC | Acc | F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| BrainWave | Chisco-R | **51.12 ± 1.56** | 2.86 ± 0.52 | 0.78 ± 0.67 | 0.06 ± 0.26 |
| BrainOmni | Chisco-R | 51.00 ± 0.67 | 3.02 ± 0.91 | 0.95 ± 0.86 | 0.17 ± 0.50 |
| LaBraM | Chisco-R | 50.94 ± 1.48 | 3.33 ± 0.68 | 1.26 ± 0.64 | 0.15 ± 0.37 |
| NeuroGPT | Chisco-R | 50.55 ± 1.12 | **4.82 ± 0.28** | 0.54 ± 0.19 | 0.16 ± 0.27 |
| CBraMod | Chisco-R | 50.51 ± 1.29 | 2.54 ± 0.89 | 1.64 ± 0.56 | -0.01 ± 0.53 |
| EEGPT | Chisco-R | 50.48 ± 0.93 | 2.58 ± 0.83 | 2.47 ± 0.85 | **0.18 ± 0.21** |
| BFM | Chisco-R | 50.40 ± 0.49 | 3.23 ± 0.60 | 1.98 ± 0.39 | 0.07 ± 0.20 |
| NeuroLM | Chisco-R | 50.30 ± 0.26 | 3.17 ± 1.51 | 0.40 ± 0.33 | -0.01 ± 0.02 |
| BIOT | Chisco-R | 50.28 ± 1.02 | 3.67 ± 0.30 | 1.62 ± 0.53 | -0.03 ± 0.27 |
| Mbrain | Chisco-R | 50.22 ± 0.37 | 3.79 ± 0.38 | 0.19 ± 0.02 | 0.00 ± 0.00 |
| REVE | Chisco-R | 50.17 ± 0.52 | 2.64 ± 0.34 | **2.59 ± 0.29** | 0.04 ± 0.33 |
| MVPFormer | Chisco-R | 50.17 ± 0.19 | 2.49 ± 0.37 | 2.54 ± 0.24 | 0.08 ± 0.25 |
| Brant | Chisco-R | 50.09 ± 0.33 | 3.81 ± 0.53 | 0.19 ± 0.03 | 0.00 ± 0.00 |
| SppEEGNet | Chisco-R | 49.98 ± 0.39 | 2.28 ± 0.34 | 2.23 ± 0.46 | 0.03 ± 0.26 |
| BrainBERT | Chisco-R | 49.91 ± 0.38 | 3.23 ± 0.53 | 0.57 ± 0.50 | 0.08 ± 0.14 |
| CodeBrain | Chisco-R | 49.91 ± 0.62 | 3.41 ± 1.07 | 1.35 ± 0.70 | -0.29 ± 0.09 |
| Bendr | Chisco-R | 49.87 ± 0.23 | 2.59 ± 0.38 | 2.59 ± 0.36 | 0.07 ± 0.34 |

| Model Name | Dataset | AUROC | Acc | F1 | Kappa |
| --- | --- | --- | --- | --- | --- |
| LaBraM | Chisco-I | **51.29 ± 0.75** | 4.02 ± 0.74 | 1.83 ± 1.27 | **0.46 ± 0.37** |
| CBraMod | Chisco-I | 50.63 ± 0.65 | 3.58 ± 0.53 | 1.82 ± 0.79 | 0.29 ± 0.31 |
| CodeBrain | Chisco-I | 50.50 ± 0.70 | **4.85 ± 0.39** | 0.63 ± 0.25 | 0.08 ± 0.13 |
| BrainOmni | Chisco-I | 50.49 ± 1.00 | 2.19 ± 0.46 | 1.63 ± 0.79 | 0.14 ± 0.14 |
| BFM | Chisco-I | 50.43 ± 0.28 | 3.20 ± 0.48 | 2.07 ± 0.31 | 0.14 ± 0.06 |
| Mbrain | Chisco-I | 50.25 ± 0.69 | 3.46 ± 0.29 | 0.23 ± 0.10 | 0.05 ± 0.11 |
| NeuroGPT | Chisco-I | 50.13 ± 0.36 | 4.72 ± 0.36 | 0.50 ± 0.14 | 0.14 ± 0.38 |
| BrainBERT | Chisco-I | 50.09 ± 0.85 | 3.25 ± 0.37 | 1.51 ± 0.89 | 0.05 ± 0.33 |
| EEGPT | Chisco-I | 50.09 ± 0.22 | 2.86 ± 0.84 | 2.41 ± 0.36 | 0.12 ± 0.29 |
| NeuroLM | Chisco-I | 50.05 ± 0.23 | 3.50 ± 0.91 | 0.38 ± 0.09 | -0.13 ± 0.14 |
| MVPFormer | Chisco-I | 50.02 ± 0.77 | 2.42 ± 0.20 | **2.62 ± 0.19** | 0.09 ± 0.18 |
| Bendr | Chisco-I | 49.97 ± 0.52 | 2.54 ± 0.15 | 2.58 ± 0.18 | 0.04 ± 0.16 |
| REVE | Chisco-I | 49.94 ± 0.45 | 2.61 ± 0.23 | 2.57 ± 0.14 | 0.00 ± 0.14 |
| SppEEGNet | Chisco-I | 49.91 ± 0.37 | 2.31 ± 0.11 | 2.23 ± 0.39 | -0.03 ± 0.12 |
| Brant | Chisco-I | 49.91 ± 0.45 | 3.54 ± 0.90 | 0.17 ± 0.04 | 0.00 ± 0.00 |
| BrainWave | Chisco-I | 49.85 ± 1.29 | 3.27 ± 1.28 | 0.80 ± 0.82 | 0.19 ± 0.25 |
| BIOT | Chisco-I | 49.30 ± 0.30 | 2.95 ± 0.99 | 1.74 ± 0.29 | 0.04 ± 0.10 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| BrainWave | HUP-ECoG | **87.93 ± 7.29** | **80.37 ± 10.90** | **72.35 ± 12.83** | **65.08 ± 17.77** |
| BrainBERT | HUP-ECoG | 81.43 ± 12.12 | 73.28 ± 8.58 | 62.80 ± 11.10 | 60.17 ± 23.04 |
| BFM | HUP-ECoG | 71.74 ± 11.02 | 71.37 ± 9.86 | 46.94 ± 13.00 | 40.89 ± 17.65 |
| CodeBrain | HUP-ECoG | 67.35 ± 4.79 | 70.51 ± 5.57 | 32.72 ± 15.89 | 30.05 ± 19.93 |
| CBraMod | HUP-ECoG | 64.37 ± 9.35 | 58.39 ± 10.39 | 38.95 ± 23.98 | 44.55 ± 30.77 |
| MVPFormer | HUP-ECoG | 59.07 ± 7.91 | 61.82 ± 6.75 | 45.67 ± 12.19 | 44.12 ± 21.72 |
| Mbrain | HUP-ECoG | 56.62 ± 8.53 | 61.01 ± 12.46 | 2.38 ± 5.32 | 1.70 ± 3.79 |
| BIOT | HUP-ECoG | 53.92 ± 8.37 | 61.48 ± 11.55 | 18.52 ± 12.72 | 40.45 ± 6.17 |
| LaBraM | HUP-ECoG | 51.45 ± 6.45 | 63.70 ± 12.51 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| SppEEGNet | HUP-ECoG | 44.10 ± 6.62 | 52.23 ± 9.47 | 34.86 ± 10.44 | 35.56 ± 9.85 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| BrainWave | HUP-SEEG | **78.57 ± 11.27** | 73.52 ± 14.26 | **49.70 ± 19.88** | **45.01 ± 25.10** |
| BFM | HUP-SEEG | 61.02 ± 2.11 | **74.15 ± 1.00** | 19.08 ± 8.05 | 13.10 ± 5.77 |
| CodeBrain | HUP-SEEG | 60.83 ± 4.10 | 69.33 ± 2.84 | 28.64 ± 9.77 | 25.36 ± 11.87 |
| BrainBERT | HUP-SEEG | 56.84 ± 8.62 | 66.63 ± 14.34 | 0.12 ± 0.28 | 0.08 ± 0.17 |
| CBraMod | HUP-SEEG | 55.40 ± 7.71 | 62.88 ± 16.93 | 37.96 ± 15.49 | 32.46 ± 12.61 |
| MVPFormer | HUP-SEEG | 54.10 ± 3.55 | 59.67 ± 18.05 | 27.07 ± 9.24 | 25.55 ± 20.13 |
| LaBraM | HUP-SEEG | 52.52 ± 8.85 | 62.51 ± 18.98 | 33.63 ± 19.24 | 29.88 ± 17.36 |
| SppEEGNet | HUP-SEEG | 51.92 ± 1.60 | 52.17 ± 18.58 | 22.26 ± 15.97 | 25.34 ± 24.34 |
| Mbrain | HUP-SEEG | 50.19 ± 7.66 | 62.03 ± 14.97 | 8.16 ± 18.24 | 12.65 ± 28.30 |
| BIOT | HUP-SEEG | 45.01 ± 8.15 | 68.24 ± 20.50 | 28.06 ± 38.92 | 30.23 ± 41.21 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| BrainWave | SWEC | **83.86 ± 7.65** | **82.04 ± 8.29** | 61.34 ± 16.62 | 58.30 ± 15.88 |
| LaBraM | SWEC | 83.02 ± 9.92 | 79.80 ± 7.12 | 59.36 ± 16.72 | 60.46 ± 20.59 |
| CBraMod | SWEC | 78.27 ± 10.72 | 74.67 ± 8.87 | **62.18 ± 4.78** | **69.72 ± 5.57** |
| BrainBERT | SWEC | 76.98 ± 8.19 | 61.41 ± 18.26 | 47.72 ± 7.41 | 54.46 ± 16.45 |
| BIOT | SWEC | 65.88 ± 5.77 | 45.30 ± 18.60 | 40.98 ± 0.95 | 55.94 ± 10.07 |
| BFM | SWEC | 65.04 ± 10.17 | 73.04 ± 3.72 | 36.43 ± 21.46 | 37.18 ± 24.27 |
| Mbrain | SWEC | 63.47 ± 4.54 | 75.01 ± 0.03 | 0.14 ± 0.30 | 0.08 ± 0.19 |
| MVPFormer | SWEC | 62.06 ± 4.81 | 63.52 ± 10.35 | 43.14 ± 5.83 | 49.47 ± 11.14 |
| CodeBrain | SWEC | 59.48 ± 7.10 | 59.85 ± 16.91 | 43.66 ± 6.51 | 51.26 ± 11.43 |
| SppEEGNet | SWEC | 44.20 ± 4.73 | 56.53 ± 10.09 | 33.53 ± 13.59 | 41.92 ± 21.34 |

| Model Name | Dataset | AUROC | Acc | F1 | F2 |
| --- | --- | --- | --- | --- | --- |
| LaBraM | Cogitate-CF | **52.95 ± 0.53** | 49.88 ± 0.07 | 53.22 ± 29.75 | 66.49 ± 37.17 |
| CodeBrain | Cogitate-CF | 52.25 ± 0.82 | **51.59 ± 1.19** | 28.72 ± 22.67 | 27.88 ± 31.59 |
| BrainBERT | Cogitate-CF | 52.14 ± 1.33 | 50.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| SppEEGNet | Cogitate-CF | 50.83 ± 1.83 | 50.37 ± 1.00 | 44.47 ± 6.59 | 42.08 ± 9.49 |
| MVPFormer | Cogitate-CF | 50.57 ± 0.50 | 48.89 ± 0.39 | 53.41 ± 5.12 | 56.81 ± 8.10 |
| Mbrain | Cogitate-CF | 50.26 ± 1.13 | 49.95 ± 0.95 | **62.95 ± 7.05** | **75.77 ± 14.67** |
| CBraMod | Cogitate-CF | 50.09 ± 1.38 | 50.03 ± 0.17 | 24.12 ± 27.84 | 25.40 ± 34.87 |
| BFM | Cogitate-CF | 49.95 ± 0.56 | 50.29 ± 0.26 | 51.32 ± 5.24 | 52.42 ± 8.55 |
| BIOT | Cogitate-CF | 49.77 ± 1.87 | 50.40 ± 0.93 | 33.72 ± 32.20 | 37.87 ± 38.74 |
| BrainWave | Cogitate-CF | 49.03 ± 1.05 | 50.00 ± 0.00 | 53.33 ± 29.81 | 66.67 ± 37.27 |
