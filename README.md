# BrainBenchmark
The benchmark of self-supervised Brain Foundation Models on electrical brain signals.

## Table of Contents
- [📋 Overview](#overview)
- [⚙️ Get Started](#start)
    * [🗄️ Data Preprocessing](#dataset)
    * [💻 Finetune and Evaluate](#finetune)
- [🪄 How to Extend](#extend)
    * [📚 Add new dataset](#newdata)
    * [🌟 Add new methods](#newmodel)
- [🎯 Benchmark Table](#result)

<h2 id="overview"> 📋 Overview </h2>

Brain4FMs is an open and extensible evaluation codebase for **Brain Foundation Models (BFMs)** on EEG/iEEG. The platform provides plug-and-play interfaces for data preprocessing, model loading, and standardized training/evaluation protocols, integrating 15 BFMs and 18 public datasets for reproducible comparisons. It is designed to be easily extended with new models, datasets, and protocols, and will be continuously updated to support frozen-encoder and few-shot evaluations, together with an open leaderboard as the field evolves.

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
python pretrained_run.py --run_mode finetune --gpu_id 0 --model LaBraM --dataset MAYO --cv_id 0 --batch_size 128
```

<h3 id="dataset"> 🗄️ Data Preprocessing </h3>

For each dataset you want to run experiments on, the first thing to do is generating a specific set of data on your device. This code provides standardized preprocessing pipelines for multiple widely-used datasets, including: [CHBMIT](https://physionet.org/content/chbmit/1.0.0/), [MAYO](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [FNUSA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [UCSD](https://openneuro.org/datasets/ds002778/versions/1.0.5), [SleepEDF](https://physionet.org/content/sleep-edfx/1.0.0/), [ISRUC](https://sleeptight.isr.uc.pt/), [Dep-122](https://doi.org/10.18112/openneuro.ds003478.v1.1.0), [SD-28](https://doi.org/10.18150/repod.0107441), [ADHD-Adult](https://doi.org/10.17632/6k4g25fhzg.1), [ADHD-Child](https://doi.org/10.21227/rzfh-zn36), [ADFD](https://openneuro.org/datasets/ds004504/versions/1.0.2), [MDD-64](https://doi.org/10.6084/m9.figshare.4244171.v2), [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html), [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/index.html), [EEGMMIDB](https://bcmi.sjtu.edu.cn/home/seed/index.html), [EEGMat](https://physionet.org/content/eegmat/1.0.0/), [BCI-2a](https://www.bbci.de/competition/iv/#dataset1), [Chisco](https://openneuro.org/datasets/ds005170/versions/1.1.2). 

Download the datasets to obtain the raw data files for your target dataset. Then, edit `config.py` under `data_preprocess/{dataset}_preprocess` to specify:
* `data_root` - Root directory where raw datasets are stored.
* `data_save_dir` - Output directory for preprocessed data.

<h3 id="finetune"> 💻 Finetune and Evaluate </h3>

This benchmark support the training and evaluating of the models with a pretrained checkpoint. You should update the path of checkpoints in `model/model_config.py`. 

### Pipeline

To load a checkpoint and train or evaluate from the checkpoint, please run the `pretrained_run.py`. For the `--run_mode` parameter, you can choose one from these strings:

- `finetune`: load the checkpoint, and begin finetune from the checkpoint.
- `test`: evaluate the model with this checkpoint.

### Note

**Fair comparison.** If you want to evaluate a result and make a direct performance comparison with other models on the same dataset, the following arguments about input data must be set according to a unified setting. These arguments includes `dataset`, `seq_len`, `patch_len`. 

**Loading checkpoints.** If you need to load ckpt (continue training from the last breakpoint), please add the `load_ckpt_path` argument (`None` if train from scratch). The path to save model checkpoints can also be set with the `save_ckpt_path` argument.

**Task MLP.** For models that do not prescribe a task-specific finetuning head, you may choose either a **Linear head** or a **CNN head** for downstream evaluation. To ensure a fair comparison across models, the benchmark reports results using the **Linear head** by default as it introduces minimal architectural bias. The architecture of the optional head can be configured with the following arguments: `cnn_in_channels`, `cnn_kernel_size` *(CNN)*.

<h2 id='extend'>🪄 How to Extend</h2>

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
   - `various_ch_num`: whether or not the channel number may varies between different data files in this dataset.
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

### Dataset
The benchmark contains 18 public datasets. 
[CHBMIT](https://physionet.org/content/chbmit/1.0.0/), [MAYO](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [FNUSA](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7297990/), [UCSD](https://openneuro.org/datasets/ds002778/versions/1.0.5), [SleepEDF](https://physionet.org/content/sleep-edfx/1.0.0/), [ISRUC](https://sleeptight.isr.uc.pt/), [Dep-122](https://doi.org/10.18112/openneuro.ds003478.v1.1.0), [Schizophrenia_28](https://doi.org/10.18150/repod.0107441), [ADHD-Adult](https://doi.org/10.17632/6k4g25fhzg.1), [ADHD-Child](https://doi.org/10.21227/rzfh-zn36), [ADFD](https://openneuro.org/datasets/ds004504/versions/1.0.2), [MD64](https://doi.org/10.6084/m9.figshare.4244171.v2), [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html), [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/index.html), [EEGMMIDB](https://bcmi.sjtu.edu.cn/home/seed/index.html), [EEGMat](https://physionet.org/content/eegmat/1.0.0/), [BCI-2a](https://www.bbci.de/competition/iv/#dataset1),[Chisco](https://openneuro.org/datasets/ds005170/versions/1.1.2).  

## Benchmark
| Mode Name | Dataset | AUROC | Acc | F1 | F2 |
| -------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| REVE | ADFD | **$84.28 \pm 7.12$** | **$77.36 \pm 7.89$** | **$79.11 \pm 6.58$** | $78.70 \pm 8.94$ |
| EEGPT | ADFD | $81.11 \pm 5.08$ | $71.89 \pm 4.86$ | $72.56 \pm 5.06$ | $69.87 \pm 7.32$ |
| BrainOmni | ADFD | $79.90 \pm 7.40$ | $71.38 \pm 4.43$ | $72.91 \pm 4.98$ | $71.90 \pm 9.42$ |
| BrainWave | ADFD | $74.11 \pm 4.67$ | $67.90 \pm 4.59$ | $68.10 \pm 9.93$ | $73.26 \pm 9.84$ |
| LaBraM | ADFD | $71.82 \pm 7.35$ | $68.03 \pm 4.81$ | $68.01 \pm 8.94$ | $71.96 \pm 7.23$ |
| BrainBERT | ADFD | $59.31 \pm 4.70$ | $58.03 \pm 4.59$ | $60.34 \pm 11.47$ | $64.04 \pm 14.65$ |
| BFM | ADFD | $58.66 \pm 8.18$ | $58.13 \pm 9.02$ | $59.45 \pm 11.51$ | $61.14 \pm 8.87$ |
| BIOT | ADFD | $55.72 \pm 17.42$ | $48.54 \pm 11.76$ | $57.58 \pm 13.67$ | $69.74 \pm 23.55$ |
| CBraMod | ADFD | $54.78 \pm 5.64$ | $58.69 \pm 3.76$ | $40.98 \pm 20.49$ | $36.82 \pm 19.93$ |
| Mbrain | ADFD | $53.46 \pm 12.08$ | $50.12 \pm 10.27$ | $52.54 \pm 12.43$ | $51.96 \pm 13.98$ |
| Bendr | ADFD | $52.08 \pm 2.12$ | $52.32 \pm 0.93$ | $53.64 \pm 4.53$ | $51.98 \pm 6.98$ |
| SppEEGNet | ADFD | $51.68 \pm 1.72$ | $50.62 \pm 1.85$ | $51.16 \pm 5.19$ | $49.09 \pm 7.11$ |
| NeuroLM | ADFD | $50.88 \pm 3.97$ | $52.77 \pm 3.44$ | $63.41 \pm 8.66$ | $71.36 \pm 15.92$ |
| NeuroGPT-E | ADFD | $50.85 \pm 2.84$ | $54.70 \pm 1.37$ | $70.57 \pm 1.18$ | $85.42 \pm 0.88$ |
| Brant1 | ADFD | $50.45 \pm 6.96$ | $53.55 \pm 3.67$ | $69.69 \pm 3.21$ | $84.23 \pm 4.00$ |
| NeuroGPT-D | ADFD | $50.04 \pm 3.34$ | $54.68 \pm 1.24$ | $70.70 \pm 1.04$ | **$85.77 \pm 0.61$** |
| EEGPT | ADHD-Adult | **$96.48 \pm 3.07$** | $91.36 \pm 3.89$ | $90.34 \pm 4.88$ | $88.85 \pm 8.18$ |
| BrainOmni | ADHD-Adult | $96.16 \pm 2.39$ | $91.12 \pm 2.38$ | $90.31 \pm 2.61$ | $89.23 \pm 4.07$ |
| BrainWave | ADHD-Adult | $96.02 \pm 2.88$ | $91.02 \pm 4.02$ | $90.59 \pm 4.10$ | $91.39 \pm 4.95$ |
| BIOT | ADHD-Adult | $95.74 \pm 2.11$ | $89.65 \pm 4.33$ | $89.14 \pm 4.45$ | $89.69 \pm 4.83$ |
| REVE | ADHD-Adult | $95.70 \pm 1.97$ | $91.46 \pm 2.33$ | $90.80 \pm 2.47$ | $90.36 \pm 4.77$ |
| Mbrain | ADHD-Adult | $95.49 \pm 3.17$ | $91.58 \pm 3.97$ | $90.96 \pm 4.33$ | $90.88 \pm 5.92$ |
| Brant1 | ADHD-Adult | $95.15 \pm 2.67$ | $89.10 \pm 3.88$ | $88.62 \pm 4.04$ | $89.53 \pm 4.67$ |
| LaBraM | ADHD-Adult | $94.95 \pm 3.59$ | $90.74 \pm 3.84$ | **$92.21 \pm 3.89$** | **$91.41 \pm 3.65$** |
| CBraMod | ADHD-Adult | $94.65 \pm 2.44$ | **$91.89 \pm 3.64$** | $91.21 \pm 4.35$ | $91.03 \pm 6.89$ |
| NeuroGPT-E | ADHD-Adult | $91.36 \pm 4.62$ | $84.28 \pm 5.65$ | $83.22 \pm 5.30$ | $82.34 \pm 2.70$ |
| BFM | ADHD-Adult | $87.90 \pm 4.47$ | $81.54 \pm 3.81$ | $80.35 \pm 4.11$ | $80.52 \pm 4.53$ |
| NeuroGPT-D | ADHD-Adult | $87.54 \pm 5.10$ | $81.96 \pm 4.69$ | $82.08 \pm 3.94$ | $85.10 \pm 3.01$ |
| NeuroLM | ADHD-Adult | $81.72 \pm 13.64$ | $74.58 \pm 6.76$ | $74.42 \pm 5.85$ | $76.80 \pm 10.58$ |
| BrainBERT | ADHD-Adult | $74.37 \pm 4.60$ | $68.83 \pm 5.20$ | $61.01 \pm 11.08$ | $56.61 \pm 13.64$ |
| Bendr | ADHD-Adult | $62.14 \pm 3.49$ | $61.33 \pm 4.30$ | $59.78 \pm 4.94$ | $60.84 \pm 6.46$ |
| SppEEGNet | ADHD-Adult | $51.32 \pm 4.02$ | $54.20 \pm 5.98$ | $35.94 \pm 29.70$ | $36.25 \pm 35.42$ |
| REVE | ADHD-Child | **$79.19 \pm 6.94$** | $71.36 \pm 4.29$ | $73.81 \pm 5.73$ | $73.51 \pm 8.21$ |
| BrainWave | ADHD-Child | $78.27 \pm 4.45$ | $70.72 \pm 3.59$ | $73.73 \pm 6.24$ | $74.49 \pm 11.98$ |
| BrainOmni | ADHD-Child | $70.83 \pm 8.81$ | $61.65 \pm 6.83$ | $62.70 \pm 8.98$ | $60.71 \pm 14.46$ |
| BFM | ADHD-Child | $69.49 \pm 11.06$ | $68.73 \pm 8.73$ | $74.95 \pm 7.31$ | $75.28 \pm 4.12$ |
| Mbrain | ADHD-Child | $69.02 \pm 10.65$ | $62.69 \pm 7.84$ | $66.45 \pm 5.50$ | $65.83 \pm 3.79$ |
| EEGPT | ADHD-Child | $67.37 \pm 13.27$ | $63.65 \pm 10.41$ | $67.59 \pm 10.29$ | $68.16 \pm 11.87$ |
| LaBraM | ADHD-Child | $65.92 \pm 12.04$ | $64.46 \pm 7.85$ | $73.52 \pm 6.01$ | $70.05 \pm 6.73$ |
| NeuroLM | ADHD-Child | $63.68 \pm 5.85$ | $60.18 \pm 4.66$ | $70.89 \pm 3.67$ | $79.79 \pm 4.83$ |
| CBraMod | ADHD-Child | $63.55 \pm 5.05$ | $63.84 \pm 5.65$ | $72.89 \pm 2.89$ | $80.65 \pm 5.03$ |
| Brant1 | ADHD-Child | $57.59 \pm 6.95$ | $54.46 \pm 6.57$ | $61.83 \pm 14.80$ | $67.99 \pm 23.07$ |
| BIOT | ADHD-Child | $55.50 \pm 5.17$ | **$74.71 \pm 5.94$** | **$85.37 \pm 3.79$** | **$92.93 \pm 1.94$** |
| NeuroGPT-E | ADHD-Child | $55.05 \pm 1.62$ | $56.26 \pm 2.50$ | $71.74 \pm 2.05$ | $86.23 \pm 1.22$ |
| BrainBERT | ADHD-Child | $55.02 \pm 3.15$ | $55.79 \pm 3.22$ | $56.08 \pm 7.40$ | $53.28 \pm 9.45$ |
| SppEEGNet | ADHD-Child | $52.92 \pm 5.04$ | $48.90 \pm 3.70$ | $47.90 \pm 6.89$ | $44.60 \pm 8.05$ |
| Bendr | ADHD-Child | $52.85 \pm 1.55$ | $50.98 \pm 1.28$ | $51.65 \pm 2.26$ | $48.77 \pm 2.39$ |
| NeuroGPT-D | ADHD-Child | $47.42 \pm 11.66$ | $55.68 \pm 6.01$ | $63.74 \pm 10.73$ | $69.26 \pm 18.07$ |
| BrainWave | CHBMIT | **$89.63 \pm 3.08$** | **$79.68 \pm 11.99$** | **$71.36 \pm 9.08$** | **$78.35 \pm 3.96$** |
| REVE | CHBMIT | $83.83 \pm 10.16$ | $78.15 \pm 12.01$ | $63.05 \pm 14.42$ | $66.43 \pm 13.25$ |
| NeuroGPT-E | CHBMIT | $78.40 \pm 11.86$ | $75.33 \pm 6.76$ | $51.35 \pm 21.65$ | $63.49 \pm 15.99$ |
| BrainBERT | CHBMIT | $78.22 \pm 6.14$ | $74.97 \pm 10.32$ | $57.59 \pm 7.80$ | $61.13 \pm 8.79$ |
| LaBraM | CHBMIT | $74.54 \pm 9.86$ | $73.25 \pm 10.50$ | $53.44 \pm 14.03$ | $57.42 \pm 17.31$ |
| BrainOmni | CHBMIT | $74.19 \pm 12.61$ | $66.81 \pm 19.09$ | $50.93 \pm 12.30$ | $56.17 \pm 11.59$ |
| BFM | CHBMIT | $73.59 \pm 4.78$ | $72.23 \pm 2.72$ | $50.57 \pm 9.01$ | $50.56 \pm 13.72$ |
| EEGPT | CHBMIT | $71.18 \pm 8.77$ | $66.70 \pm 10.91$ | $45.85 \pm 5.99$ | $51.21 \pm 14.56$ |
| Mbrain | CHBMIT | $71.04 \pm 3.32$ | $73.04 \pm 5.88$ | $30.76 \pm 14.98$ | $28.46 \pm 19.77$ |
| CBraMod | CHBMIT | $69.36 \pm 4.03$ | $70.33 \pm 8.40$ | $45.99 \pm 7.69$ | $48.74 \pm 13.97$ |
| NeuroLM | CHBMIT | $66.86 \pm 6.28$ | $53.52 \pm 19.85$ | $26.79 \pm 22.54$ | $38.06 \pm 33.59$ |
| NeuroGPT-D | CHBMIT | $61.00 \pm 7.27$ | $72.51 \pm 3.11$ | $18.36 \pm 23.45$ | $19.12 \pm 25.08$ |
| BIOT | CHBMIT | $55.55 \pm 8.49$ | $28.57 \pm 5.29$ | $41.39 \pm 2.64$ | $61.44 \pm 2.12$ |
| Brant1 | CHBMIT | $55.07 \pm 5.47$ | $69.64 \pm 0.57$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ |
| Bendr | CHBMIT | $54.51 \pm 3.12$ | $59.46 \pm 3.49$ | $32.07 \pm 1.73$ | $35.55 \pm 3.69$ |
| SppEEGNet | CHBMIT | $43.48 \pm 4.87$ | $60.48 \pm 5.90$ | $31.21 \pm 3.71$ | $33.12 \pm 5.21$ |
| BrainWave | Dep-BDI | **$72.40 \pm 3.48$** | $66.03 \pm 3.03$ | **$61.80 \pm 2.87$** | $67.47 \pm 5.82$ |
| BrainOmni | Dep-BDI | $67.58 \pm 12.08$ | $62.70 \pm 8.95$ | $54.94 \pm 9.86$ | $58.32 \pm 13.18$ |
| REVE | Dep-BDI | $67.07 \pm 12.18$ | $64.58 \pm 7.59$ | $47.14 \pm 14.04$ | $44.91 \pm 16.75$ |
| LaBraM | Dep-BDI | $64.73 \pm 10.38$ | $58.98 \pm 10.86$ | $59.91 \pm 9.04$ | **$70.65 \pm 8.26$** |
| BrainBERT | Dep-BDI | $63.70 \pm 8.81$ | $61.28 \pm 5.16$ | $47.77 \pm 11.05$ | $47.91 \pm 12.24$ |
| EEGPT | Dep-BDI | $62.57 \pm 4.25$ | $61.68 \pm 5.26$ | $50.03 \pm 5.10$ | $51.01 \pm 10.94$ |
| BFM | Dep-BDI | $62.32 \pm 10.19$ | $58.79 \pm 6.39$ | $48.40 \pm 10.28$ | $50.73 \pm 12.92$ |
| BIOT | Dep-BDI | $59.90 \pm 10.70$ | $29.76 \pm 17.22$ | $11.04 \pm 2.01$ | $21.86 \pm 4.56$ |
| Mbrain | Dep-BDI | $59.36 \pm 6.63$ | $62.44 \pm 2.82$ | $24.48 \pm 16.42$ | $20.45 \pm 15.75$ |
| NeuroGPT-E | Dep-BDI | $59.28 \pm 8.17$ | $62.36 \pm 1.18$ | $1.69 \pm 1.73$ | $1.08 \pm 1.10$ |
| NeuroLM | Dep-BDI | $54.36 \pm 4.74$ | $61.06 \pm 2.32$ | $29.95 \pm 25.82$ | $30.76 \pm 31.38$ |
| Bendr | Dep-BDI | $54.18 \pm 3.12$ | $54.47 \pm 2.65$ | $44.92 \pm 2.87$ | $47.47 \pm 3.06$ |
| CBraMod | Dep-BDI | $53.57 \pm 5.15$ | **$68.96 \pm 14.03$** | $12.59 \pm 7.85$ | $16.05 \pm 9.63$ |
| Brant1 | Dep-BDI | $52.56 \pm 8.72$ | $62.39 \pm 1.21$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ |
| NeuroGPT-D | Dep-BDI | $51.42 \pm 8.06$ | $62.39 \pm 1.21$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ |
| SppEEGNet | Dep-BDI | $49.61 \pm 2.42$ | $48.69 \pm 3.83$ | $41.44 \pm 3.60$ | $45.27 \pm 3.89$ |
| REVE | EEGMat | **$77.95 \pm 9.10$** | **$75.41 \pm 7.90$** | $39.49 \pm 26.69$ | $60.60 \pm 8.59$ |
| LaBraM | EEGMat | $70.51 \pm 3.12$ | $67.71 \pm 4.12$ | $47.82 \pm 7.14$ | $52.83 \pm 12.87$ |
| NeuroGPT-E | EEGMat | $70.25 \pm 3.60$ | $73.44 \pm 3.75$ | $24.83 \pm 20.49$ | $22.49 \pm 19.77$ |
| Mbrain | EEGMat | $68.45 \pm 6.03$ | $74.90 \pm 2.44$ | $12.68 \pm 14.64$ | $9.63 \pm 11.19$ |
| BrainOmni | EEGMat | $68.22 \pm 7.40$ | $59.55 \pm 6.57$ | $47.51 \pm 3.87$ | $59.59 \pm 6.03$ |
| EEGPT | EEGMat | $67.48 \pm 3.61$ | $63.64 \pm 6.38$ | **$49.15 \pm 2.00$** | $59.16 \pm 3.60$ |
| BrainWave | EEGMat | $66.33 \pm 6.38$ | $51.48 \pm 20.10$ | $43.61 \pm 6.91$ | $54.67 \pm 8.48$ |
| CBraMod | EEGMat | $63.47 \pm 6.36$ | $72.11 \pm 4.05$ | $19.31 \pm 18.02$ | $16.27 \pm 16.92$ |
| NeuroLM | EEGMat | $63.23 \pm 8.05$ | $31.20 \pm 10.30$ | $42.14 \pm 4.49$ | $63.60 \pm 4.31$ |
| BrainBERT | EEGMat | $61.42 \pm 7.32$ | $66.73 \pm 6.27$ | $44.56 \pm 8.65$ | $49.56 \pm 12.95$ |
| BFM | EEGMat | $59.98 \pm 2.71$ | $65.79 \pm 4.27$ | $34.96 \pm 5.56$ | $36.22 \pm 9.89$ |
| BIOT | EEGMat | $59.41 \pm 7.70$ | $26.69 \pm 0.59$ | $42.13 \pm 0.73$ | **$64.54 \pm 0.68$** |
| Brant1 | EEGMat | $57.13 \pm 12.45$ | $74.67 \pm 0.59$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ |
| Bendr | EEGMat | $53.69 \pm 1.95$ | $65.83 \pm 4.95$ | $34.13 \pm 6.75$ | $34.47 \pm 6.33$ |
| SppEEGNet | EEGMat | $52.32 \pm 3.48$ | $49.70 \pm 7.33$ | $36.94 \pm 4.12$ | $47.43 \pm 8.97$ |
| NeuroGPT-D | EEGMat | $51.04 \pm 5.91$ | $73.14 \pm 0.64$ | $0.00 \pm 0.00$ | $0.00 \pm 0.00$ |
| BrainWave | FNUSA | **$92.46 \pm 4.66$** | $88.68 \pm 6.28$ | **$82.69 \pm 6.13$** | **$82.97 \pm 3.97$** |
| Mbrain | FNUSA | $91.00 \pm 8.23$ | $86.72 \pm 8.07$ | $75.20 \pm 12.91$ | $75.51 \pm 16.22$ |
| EEGPT | FNUSA | $90.45 \pm 4.49$ | $84.35 \pm 5.97$ | $74.63 \pm 10.75$ | $77.85 \pm 7.86$ |
| NeuroGPT-E | FNUSA | $90.21 \pm 6.26$ | $85.79 \pm 5.39$ | $74.50 \pm 4.13$ | $72.47 \pm 10.94$ |
| BrainBERT | FNUSA | $90.18 \pm 3.81$ | **$89.04 \pm 4.27$** | $77.46 \pm 8.73$ | $79.24 \pm 8.11$ |
| LaBraM | FNUSA | $89.27 \pm 7.54$ | $81.66 \pm 9.79$ | $71.86 \pm 14.62$ | $75.93 \pm 11.34$ |
| BrainOmni | FNUSA | $88.16 \pm 6.59$ | $83.59 \pm 5.34$ | $72.55 \pm 8.33$ | $74.59 \pm 8.51$ |
| Bendr | FNUSA | $88.05 \pm 6.11$ | $83.61 \pm 5.70$ | $73.26 \pm 11.44$ | $76.76 \pm 9.99$ |
| REVE | FNUSA | $87.08 \pm 7.73$ | $82.15 \pm 6.76$ | $71.19 \pm 10.10$ | $73.81 \pm 8.46$ |
| BIOT | FNUSA | $87.05 \pm 6.69$ | $82.90 \pm 8.42$ | $73.38 \pm 13.99$ | $77.34 \pm 9.83$ |
| Brant1 | FNUSA | $86.95 \pm 12.35$ | $83.73 \pm 6.78$ | $74.71 \pm 6.32$ | $78.89 \pm 8.28$ |
| BFM | FNUSA | $77.79 \pm 12.49$ | $69.22 \pm 11.08$ | $62.03 \pm 5.26$ | $69.22 \pm 8.25$ |
| NeuroGPT-D | FNUSA | $71.91 \pm 12.37$ | $76.54 \pm 12.12$ | $29.71 \pm 37.60$ | $26.95 \pm 34.89$ |
| CBraMod | FNUSA | $71.20 \pm 13.14$ | $77.79 \pm 7.85$ | $62.46 \pm 13.02$ | $63.53 \pm 11.62$ |
| SppEEGNet | FNUSA | $64.19 \pm 7.74$ | $66.79 \pm 5.33$ | $47.70 \pm 12.81$ | $52.29 \pm 18.30$ |
| NeuroLM | FNUSA | $64.05 \pm 14.38$ | $71.52 \pm 16.80$ | $37.35 \pm 27.71$ | $37.82 \pm 35.05$ |
| BrainWave | MAYO | **$97.72 \pm 0.99$** | $93.04 \pm 2.29$ | $80.97 \pm 8.79$ | **$86.32 \pm 3.37$** |
| BrainBERT | MAYO | $96.97 \pm 0.97$ | **$94.97 \pm 0.86$** | **$81.17 \pm 6.86$** | $82.54 \pm 6.53$ |
| LaBraM | MAYO | $95.99 \pm 1.83$ | $92.98 \pm 2.41$ | $75.61 \pm 11.99$ | $80.36 \pm 8.70$ |
| NeuroGPT-E | MAYO | $95.97 \pm 2.02$ | $93.06 \pm 1.59$ | $71.47 \pm 6.56$ | $66.58 \pm 7.84$ |
| Bendr | MAYO | $92.50 \pm 3.18$ | $90.49 \pm 2.38$ | $68.82 \pm 12.60$ | $75.34 \pm 11.21$ |
| Mbrain | MAYO | $92.32 \pm 4.11$ | $91.64 \pm 1.90$ | $70.45 \pm 10.69$ | $73.39 \pm 8.72$ |
| REVE | MAYO | $92.08 \pm 3.75$ | $89.42 \pm 3.56$ | $66.17 \pm 13.22$ | $71.92 \pm 11.09$ |
| Brant1 | MAYO | $92.08 \pm 3.36$ | $81.84 \pm 12.31$ | $58.48 \pm 18.68$ | $69.02 \pm 13.70$ |
| EEGPT | MAYO | $91.76 \pm 3.16$ | $90.16 \pm 2.64$ | $66.91 \pm 10.34$ | $70.04 \pm 6.82$ |
| BrainOmni | MAYO | $91.40 \pm 6.06$ | $90.80 \pm 2.20$ | $66.93 \pm 12.71$ | $68.95 \pm 12.07$ |
| BIOT | MAYO | $89.65 \pm 7.20$ | $88.49 \pm 5.05$ | $62.95 \pm 14.38$ | $72.39 \pm 13.37$ |
| CBraMod | MAYO | $89.14 \pm 3.73$ | $87.20 \pm 1.85$ | $58.76 \pm 12.13$ | $63.80 \pm 10.13$ |
| BFM | MAYO | $81.28 \pm 7.99$ | $68.35 \pm 7.73$ | $42.56 \pm 14.69$ | $58.78 \pm 13.92$ |
| NeuroGPT-D | MAYO | $77.14 \pm 14.89$ | $87.82 \pm 3.19$ | $23.12 \pm 26.31$ | $18.52 \pm 21.13$ |
| NeuroLM | MAYO | $63.75 \pm 9.25$ | $72.21 \pm 15.86$ | $30.18 \pm 20.44$ | $37.23 \pm 25.89$ |
| SppEEGNet | MAYO | $56.07 \pm 3.10$ | $74.90 \pm 5.32$ | $33.53 \pm 8.67$ | $39.11 \pm 8.60$ |
| REVE | MDD-64 | **$96.88 \pm 4.33$** | **$88.49 \pm 8.50$** | **$90.13 \pm 6.47$** | **$92.42 \pm 4.10$** |
| EEGPT | MDD-64 | $94.38 \pm 5.39$ | $86.91 \pm 7.61$ | $88.39 \pm 6.24$ | $90.08 \pm 4.44$ |
| BrainOmni | MDD-64 | $93.89 \pm 6.98$ | $84.97 \pm 9.05$ | $87.00 \pm 7.39$ | $89.37 \pm 4.76$ |
| BrainWave | MDD-64 | $93.52 \pm 2.65$ | $85.21 \pm 1.75$ | $83.54 \pm 3.06$ | $80.53 \pm 5.66$ |
| Mbrain | MDD-64 | $93.20 \pm 7.79$ | $86.92 \pm 10.60$ | $88.94 \pm 7.62$ | $91.21 \pm 4.99$ |
| BFM | MDD-64 | $92.56 \pm 7.80$ | $86.22 \pm 10.63$ | $88.37 \pm 7.58$ | $90.93 \pm 4.78$ |
| BrainBERT | MDD-64 | $91.98 \pm 7.05$ | $84.71 \pm 7.10$ | $85.38 \pm 6.40$ | $84.84 \pm 10.13$ |
| Brant1 | MDD-64 | $91.57 \pm 9.10$ | $79.69 \pm 10.35$ | $79.70 \pm 13.60$ | $79.96 \pm 19.39$ |
| CBraMod | MDD-64 | $89.71 \pm 12.75$ | $84.11 \pm 11.79$ | $85.76 \pm 9.24$ | $86.54 \pm 8.71$ |
| LaBraM | MDD-64 | $87.22 \pm 11.73$ | $81.04 \pm 10.79$ | $78.02 \pm 8.69$ | $82.91 \pm 3.95$ |
| NeuroGPT-E | MDD-64 | $87.01 \pm 5.23$ | $80.18 \pm 5.39$ | $83.75 \pm 3.30$ | $90.08 \pm 3.35$ |
| NeuroLM | MDD-64 | $82.39 \pm 10.61$ | $80.08 \pm 6.25$ | $80.42 \pm 9.06$ | $81.15 \pm 17.53$ |
| Bendr | MDD-64 | $77.57 \pm 6.58$ | $73.48 \pm 4.41$ | $75.36 \pm 2.97$ | $75.60 \pm 4.02$ |
| NeuroGPT-D | MDD-64 | $69.90 \pm 12.45$ | $67.03 \pm 13.47$ | $76.92 \pm 7.42$ | $89.05 \pm 3.91$ |
| BIOT | MDD-64 | $52.54 \pm 12.50$ | $33.05 \pm 0.53$ | $49.54 \pm 0.54$ | $70.95 \pm 0.36$ |
| SppEEGNet | MDD-64 | $52.31 \pm 2.45$ | $54.57 \pm 3.75$ | $49.92 \pm 5.10$ | $45.27 \pm 4.94$ |
| BrainWave | SD-28 | **$88.16 \pm 7.45$** | **$81.38 \pm 6.10$** | **$83.21 \pm 8.42$** | $81.13 \pm 11.12$ |
| REVE | SD-28 | $81.01 \pm 12.95$ | $75.39 \pm 12.76$ | $79.78 \pm 9.78$ | **$85.69 \pm 11.01$** |
| NeuroGPT-D | SD-28 | $80.49 \pm 16.78$ | $76.52 \pm 13.49$ | $79.85 \pm 10.90$ | $83.84 \pm 13.56$ |
| BrainBERT | SD-28 | $72.84 \pm 11.26$ | $65.99 \pm 15.43$ | $62.55 \pm 25.88$ | $61.59 \pm 28.72$ |
| NeuroGPT-E | SD-28 | $71.76 \pm 11.67$ | $66.11 \pm 14.10$ | $72.81 \pm 14.57$ | $68.67 \pm 18.87$ |
| BrainOmni | SD-28 | $69.33 \pm 14.96$ | $69.28 \pm 12.09$ | $72.54 \pm 13.62$ | $76.60 \pm 18.78$ |
| BFM | SD-28 | $69.01 \pm 14.87$ | $62.29 \pm 4.73$ | $72.95 \pm 3.72$ | $80.97 \pm 7.70$ |
| Mbrain | SD-28 | $57.93 \pm 9.76$ | $58.33 \pm 10.29$ | $64.19 \pm 12.59$ | $66.29 \pm 15.05$ |
| EEGPT | SD-28 | $56.15 \pm 21.40$ | $55.75 \pm 10.99$ | $62.02 \pm 11.74$ | $66.06 \pm 15.41$ |
| BIOT | SD-28 | $56.10 \pm 8.70$ | $61.62 \pm 12.39$ | $67.54 \pm 18.41$ | $65.14 \pm 20.56$ |
| CBraMod | SD-28 | $54.36 \pm 13.85$ | $58.34 \pm 4.62$ | $57.87 \pm 17.84$ | $57.73 \pm 23.89$ |
| LaBraM | SD-28 | $54.12 \pm 7.16$ | $54.46 \pm 3.65$ | $59.28 \pm 6.78$ | $58.98 \pm 12.64$ |
| Brant1 | SD-28 | $50.53 \pm 11.47$ | $46.05 \pm 8.49$ | $46.67 \pm 17.48$ | $50.19 \pm 29.78$ |
| Bendr | SD-28 | $49.49 \pm 4.53$ | $49.82 \pm 3.40$ | $48.24 \pm 4.64$ | $45.26 \pm 4.01$ |
| SppEEGNet | SD-28 | $48.81 \pm 2.98$ | $47.12 \pm 3.54$ | $32.75 \pm 6.19$ | $26.81 \pm 4.89$ |
| NeuroLM | SD-28 | $48.61 \pm 4.39$ | $49.44 \pm 4.58$ | $52.88 \pm 28.48$ | $62.40 \pm 35.05$ |
| BrainOmni | UCSD-OFF | **$63.57 \pm 8.57$** | $57.37 \pm 6.38$ | $54.35 \pm 15.76$ | $56.84 \pm 26.36$ |
| CBraMod | UCSD-OFF | $62.65 \pm 5.36$ | $57.73 \pm 4.05$ | $45.62 \pm 8.78$ | $40.13 \pm 11.56$ |
| REVE | UCSD-OFF | $62.49 \pm 19.12$ | **$58.23 \pm 17.19$** | $58.00 \pm 18.89$ | $59.96 \pm 24.16$ |
| EEGPT | UCSD-OFF | $60.74 \pm 12.55$ | $55.29 \pm 7.61$ | $55.86 \pm 10.18$ | $57.90 \pm 17.09$ |
| LaBraM | UCSD-OFF | $59.14 \pm 5.09$ | $56.49 \pm 3.36$ | $57.29 \pm 8.08$ | $59.72 \pm 15.06$ |
| BrainWave | UCSD-OFF | $57.98 \pm 5.97$ | $53.03 \pm 9.54$ | **$61.67 \pm 12.08$** | $70.80 \pm 18.97$ |
| NeuroGPT-E | UCSD-OFF | $56.16 \pm 6.47$ | $53.69 \pm 5.27$ | $56.55 \pm 12.15$ | $61.68 \pm 21.55$ |
| BFM | UCSD-OFF | $53.33 \pm 3.38$ | $48.81 \pm 12.09$ | $35.87 \pm 9.19$ | $46.51 \pm 17.11$ |
| SppEEGNet | UCSD-OFF | $53.12 \pm 6.80$ | $50.58 \pm 5.85$ | $59.49 \pm 14.54$ | **$71.24 \pm 25.21$** |
| Bendr | UCSD-OFF | $52.46 \pm 4.10$ | $50.91 \pm 2.75$ | $48.03 \pm 4.59$ | $46.86 \pm 4.91$ |
| Brant1 | UCSD-OFF | $51.79 \pm 9.73$ | $44.67 \pm 21.92$ | $42.33 \pm 28.89$ | $57.52 \pm 34.82$ |
| BrainBERT | UCSD-OFF | $51.23 \pm 6.25$ | $54.50 \pm 5.67$ | $49.36 \pm 16.71$ | $49.32 \pm 20.33$ |
| NeuroLM | UCSD-OFF | $50.65 \pm 6.91$ | $50.15 \pm 4.99$ | $53.12 \pm 15.12$ | $60.97 \pm 22.89$ |
| BIOT | UCSD-OFF | $50.52 \pm 18.51$ | $49.48 \pm 4.92$ | $55.87 \pm 26.97$ | $68.51 \pm 34.83$ |
| Mbrain | UCSD-OFF | $50.40 \pm 4.10$ | $43.52 \pm 3.69$ | $31.68 \pm 23.15$ | $33.82 \pm 28.96$ |
| NeuroGPT-D | UCSD-OFF | $44.78 \pm 7.23$ | $50.64 \pm 4.60$ | $0.58 \pm 1.30$ | $0.37 \pm 0.82$ |
| BrainWave | UCSD-ON | **$68.71 \pm 19.67$** | $52.60 \pm 7.68$ | $60.48 \pm 6.06$ | $68.79 \pm 14.18$ |
| REVE | UCSD-ON | $62.81 \pm 18.92$ | **$55.87 \pm 13.27$** | $44.78 \pm 28.58$ | $45.61 \pm 33.20$ |
| CBraMod | UCSD-ON | $60.04 \pm 11.73$ | $48.98 \pm 8.00$ | $18.11 \pm 8.27$ | $13.48 \pm 6.77$ |
| NeuroGPT-E | UCSD-ON | $55.83 \pm 10.18$ | $53.05 \pm 7.10$ | $58.00 \pm 10.48$ | $63.53 \pm 13.45$ |
| Brant1 | UCSD-ON | $55.77 \pm 11.46$ | $51.94 \pm 10.63$ | $28.79 \pm 27.51$ | $31.10 \pm 33.85$ |
| BrainOmni | UCSD-ON | $53.19 \pm 14.34$ | $49.34 \pm 11.92$ | $49.03 \pm 16.01$ | $50.86 \pm 20.32$ |
| EEGPT | UCSD-ON | $51.12 \pm 21.42$ | $48.47 \pm 17.17$ | $49.21 \pm 14.90$ | $49.40 \pm 13.73$ |
| Bendr | UCSD-ON | $51.12 \pm 4.74$ | $48.64 \pm 4.70$ | $43.15 \pm 4.96$ | $41.08 \pm 6.35$ |
| LaBraM | UCSD-ON | $50.67 \pm 21.83$ | $45.18 \pm 10.52$ | $32.57 \pm 19.27$ | $31.49 \pm 19.10$ |
| BrainBERT | UCSD-ON | $50.61 \pm 4.84$ | $50.57 \pm 6.29$ | $51.39 \pm 11.05$ | $53.78 \pm 16.73$ |
| SppEEGNet | UCSD-ON | $48.84 \pm 8.07$ | $49.92 \pm 10.58$ | $45.56 \pm 22.87$ | $48.39 \pm 27.06$ |
| Mbrain | UCSD-ON | $48.83 \pm 15.48$ | $46.37 \pm 13.64$ | $36.57 \pm 17.26$ | $34.19 \pm 17.78$ |
| BIOT | UCSD-ON | $46.12 \pm 18.33$ | $48.77 \pm 4.54$ | **$64.55 \pm 4.85$** | **$81.37 \pm 3.90$** |
| NeuroGPT-D | UCSD-ON | $45.97 \pm 9.87$ | $46.78 \pm 5.61$ | $28.80 \pm 24.98$ | $30.60 \pm 28.21$ |
| NeuroLM | UCSD-ON | $44.03 \pm 17.64$ | $50.59 \pm 17.60$ | $46.84 \pm 24.42$ | $48.30 \pm 28.51$ |
| BFM | UCSD-ON | $43.66 \pm 8.29$ | $48.45 \pm 6.83$ | $42.21 \pm 17.00$ | $41.72 \pm 19.84$ |


| Mode Name | Dataset | AUROC | Acc | F1 | Kappa |
| -------------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| REVE | BCI-2a | **$61.39 \pm 1.43$** | $33.59 \pm 0.87$ | **$35.69 \pm 1.10$** | $11.46 \pm 1.16$ |
| NeuroGPT-E | BCI-2a | $60.68 \pm 2.37$ | **$34.59 \pm 3.05$** | $34.78 \pm 3.56$ | **$12.79 \pm 4.07$** |
| EEGPT | BCI-2a | $56.73 \pm 3.50$ | $27.69 \pm 1.76$ | $26.69 \pm 2.89$ | $3.62 \pm 2.41$ |
| LaBraM | BCI-2a | $54.66 \pm 2.57$ | $28.20 \pm 2.31$ | $26.78 \pm 5.86$ | $4.52 \pm 2.69$ |
| Brant1 | BCI-2a | $54.12 \pm 0.49$ | $25.00 \pm 0.00$ | $10.00 \pm 0.00$ | $0.00 \pm 0.00$ |
| BrainWave | BCI-2a | $53.98 \pm 2.34$ | $27.34 \pm 1.66$ | $25.23 \pm 6.27$ | $3.12 \pm 2.22$ |
| Mbrain | BCI-2a | $53.85 \pm 1.77$ | $27.04 \pm 1.71$ | $21.02 \pm 5.24$ | $2.72 \pm 2.28$ |
| BrainOmni | BCI-2a | $53.80 \pm 1.11$ | $26.59 \pm 2.18$ | $26.97 \pm 2.37$ | $2.12 \pm 2.91$ |
| CBraMod | BCI-2a | $53.68 \pm 1.45$ | $26.65 \pm 0.88$ | $27.62 \pm 1.78$ | $2.20 \pm 1.17$ |
| NeuroLM | BCI-2a | $51.87 \pm 1.75$ | $27.23 \pm 1.32$ | $24.58 \pm 6.26$ | $2.97 \pm 1.76$ |
| BFM | BCI-2a | $51.24 \pm 0.61$ | $25.98 \pm 0.87$ | $25.91 \pm 0.77$ | $1.31 \pm 1.15$ |
| NeuroGPT-D | BCI-2a | $51.20 \pm 2.83$ | $25.55 \pm 1.42$ | $25.50 \pm 4.91$ | $0.73 \pm 1.89$ |
| Bendr | BCI-2a | $50.96 \pm 0.63$ | $25.38 \pm 0.80$ | $25.39 \pm 0.75$ | $0.50 \pm 1.07$ |
| SppEEGNet | BCI-2a | $50.51 \pm 0.55$ | $25.23 \pm 1.49$ | $24.96 \pm 1.65$ | $0.31 \pm 1.99$ |
| BrainBERT | BCI-2a | $49.96 \pm 0.98$ | $25.35 \pm 1.73$ | $25.51 \pm 2.65$ | $0.46 \pm 2.31$ |
| BIOT | BCI-2a | $49.90 \pm 1.33$ | $23.86 \pm 1.66$ | $21.93 \pm 2.79$ | $-1.52 \pm 2.22$ |
| LaBraM | Chisco-I | **$51.29 \pm 0.75$** | $4.02 \pm 0.74$ | $1.83 \pm 1.27$ | **$0.46 \pm 0.37$** |
| CBraMod | Chisco-I | $50.63 \pm 0.65$ | $3.58 \pm 0.53$ | $1.82 \pm 0.79$ | $0.29 \pm 0.31$ |
| BrainOmni | Chisco-I | $50.49 \pm 1.00$ | $2.19 \pm 0.46$ | $1.63 \pm 0.79$ | $0.14 \pm 0.14$ |
| BFM | Chisco-I | $50.43 \pm 0.28$ | $3.20 \pm 0.48$ | $2.07 \pm 0.31$ | $0.14 \pm 0.06$ |
| Mbrain | Chisco-I | $50.25 \pm 0.69$ | $3.46 \pm 0.29$ | $0.23 \pm 0.10$ | $0.05 \pm 0.11$ |
| NeuroGPT-D | Chisco-I | $50.13 \pm 0.36$ | $4.72 \pm 0.36$ | $0.50 \pm 0.14$ | $0.14 \pm 0.38$ |
| BrainBERT | Chisco-I | $50.09 \pm 0.85$ | $3.25 \pm 0.37$ | $1.51 \pm 0.89$ | $0.05 \pm 0.33$ |
| EEGPT | Chisco-I | $50.09 \pm 0.22$ | $2.86 \pm 0.84$ | $2.41 \pm 0.36$ | $0.12 \pm 0.29$ |
| NeuroLM | Chisco-I | $50.05 \pm 0.23$ | $3.50 \pm 0.91$ | $0.38 \pm 0.09$ | $-0.13 \pm 0.14$ |
| Bendr | Chisco-I | $49.97 \pm 0.52$ | $2.54 \pm 0.15$ | **$2.58 \pm 0.18$** | $0.04 \pm 0.16$ |
| REVE | Chisco-I | $49.94 \pm 0.45$ | $2.61 \pm 0.23$ | $2.57 \pm 0.14$ | $0.00 \pm 0.14$ |
| SppEEGNet | Chisco-I | $49.91 \pm 0.37$ | $2.31 \pm 0.11$ | $2.23 \pm 0.39$ | $-0.03 \pm 0.12$ |
| Brant1 | Chisco-I | $49.88 \pm 0.55$ | $3.91 \pm 0.86$ | $0.15 \pm 0.01$ | $0.00 \pm 0.00$ |
| NeuroGPT-E | Chisco-I | $49.88 \pm 0.78$ | **$4.94 \pm 0.42$** | $0.58 \pm 0.49$ | $-0.02 \pm 0.39$ |
| BrainWave | Chisco-I | $49.85 \pm 1.29$ | $3.27 \pm 1.28$ | $0.80 \pm 0.82$ | $0.19 \pm 0.25$ |
| BIOT | Chisco-I | $49.30 \pm 0.30$ | $2.95 \pm 0.99$ | $1.74 \pm 0.29$ | $0.04 \pm 0.10$ |
| BrainWave | Chisco-R | **$51.12 \pm 1.56$** | $2.86 \pm 0.52$ | $0.78 \pm 0.67$ | $0.06 \pm 0.26$ |
| BrainOmni | Chisco-R | $51.00 \pm 0.67$ | $3.02 \pm 0.91$ | $0.95 \pm 0.86$ | $0.17 \pm 0.50$ |
| LaBraM | Chisco-R | $50.94 \pm 1.48$ | $3.33 \pm 0.68$ | $1.26 \pm 0.64$ | $0.15 \pm 0.37$ |
| NeuroGPT-D | Chisco-R | $50.55 \pm 1.12$ | **$4.82 \pm 0.28$** | $0.54 \pm 0.19$ | $0.16 \pm 0.27$ |
| CBraMod | Chisco-R | $50.51 \pm 1.29$ | $2.54 \pm 0.89$ | $1.64 \pm 0.56$ | $-0.01 \pm 0.53$ |
| EEGPT | Chisco-R | $50.48 \pm 0.93$ | $2.58 \pm 0.83$ | $2.47 \pm 0.85$ | **$0.18 \pm 0.21$** |
| NeuroGPT-E | Chisco-R | $50.45 \pm 1.00$ | $4.70 \pm 0.38$ | $0.54 \pm 0.19$ | $0.09 \pm 0.29$ |
| BFM | Chisco-R | $50.40 \pm 0.49$ | $3.23 \pm 0.60$ | $1.98 \pm 0.39$ | $0.07 \pm 0.20$ |
| NeuroLM | Chisco-R | $50.30 \pm 0.26$ | $3.17 \pm 1.51$ | $0.40 \pm 0.33$ | $-0.01 \pm 0.02$ |
| BIOT | Chisco-R | $50.28 \pm 1.02$ | $3.67 \pm 0.30$ | $1.62 \pm 0.53$ | $-0.03 \pm 0.27$ |
| Mbrain | Chisco-R | $50.22 \pm 0.37$ | $3.79 \pm 0.38$ | $0.19 \pm 0.02$ | $0.00 \pm 0.00$ |
| REVE | Chisco-R | $50.17 \pm 0.52$ | $2.64 \pm 0.34$ | **$2.59 \pm 0.29$** | $0.04 \pm 0.33$ |
| Brant1 | Chisco-R | $50.16 \pm 0.47$ | $4.24 \pm 0.84$ | $0.19 \pm 0.07$ | $0.00 \pm 0.00$ |
| SppEEGNet | Chisco-R | $49.98 \pm 0.39$ | $2.28 \pm 0.34$ | $2.23 \pm 0.46$ | $0.03 \pm 0.26$ |
| BrainBERT | Chisco-R | $49.91 \pm 0.38$ | $3.23 \pm 0.53$ | $0.57 \pm 0.50$ | $0.08 \pm 0.14$ |
| Bendr | Chisco-R | $49.87 \pm 0.23$ | $2.59 \pm 0.38$ | $2.59 \pm 0.36$ | $0.07 \pm 0.34$ |
| BrainBERT | DEAP | **$52.02 \pm 2.12$** | $22.97 \pm 2.24$ | $26.70 \pm 2.37$ | $1.73 \pm 2.01$ |
| SppEEGNet | DEAP | $51.35 \pm 2.85$ | $26.78 \pm 5.76$ | $25.80 \pm 2.11$ | $0.72 \pm 2.69$ |
| NeuroGPT-D | DEAP | $51.34 \pm 3.14$ | **$43.59 \pm 3.62$** | $15.16 \pm 0.90$ | $0.00 \pm 0.00$ |
| BrainWave | DEAP | $51.26 \pm 2.66$ | $27.11 \pm 6.76$ | $26.10 \pm 5.49$ | $1.05 \pm 7.71$ |
| CBraMod | DEAP | $51.04 \pm 0.64$ | $18.76 \pm 1.88$ | $24.58 \pm 5.09$ | $-0.14 \pm 0.87$ |
| Mbrain | DEAP | $50.75 \pm 4.59$ | $41.79 \pm 4.16$ | $14.71 \pm 1.03$ | $0.00 \pm 0.00$ |
| NeuroLM | DEAP | $50.70 \pm 3.42$ | $33.34 \pm 13.49$ | $15.26 \pm 3.62$ | $-0.26 \pm 0.70$ |
| LaBraM | DEAP | $50.57 \pm 2.03$ | $28.77 \pm 4.48$ | $25.74 \pm 2.97$ | $1.61 \pm 1.42$ |
| BIOT | DEAP | $50.29 \pm 3.34$ | $30.06 \pm 5.76$ | $22.96 \pm 4.50$ | $-1.39 \pm 2.98$ |
| REVE | DEAP | $50.24 \pm 5.00$ | $26.17 \pm 5.38$ | $24.37 \pm 5.03$ | $-1.26 \pm 3.48$ |
| Bendr | DEAP | $50.18 \pm 1.84$ | $26.35 \pm 2.09$ | $25.61 \pm 2.44$ | $-0.37 \pm 3.49$ |
| EEGPT | DEAP | $49.83 \pm 1.76$ | $26.47 \pm 5.33$ | **$26.70 \pm 4.89$** | **$2.89 \pm 2.70$** |
| BFM | DEAP | $49.57 \pm 2.04$ | $28.20 \pm 6.66$ | $24.83 \pm 1.10$ | $0.49 \pm 1.46$ |
| NeuroGPT-E | DEAP | $48.71 \pm 1.03$ | $41.96 \pm 3.03$ | $17.95 \pm 4.05$ | $-0.44 \pm 0.63$ |
| Brant1 | DEAP | $48.41 \pm 2.27$ | $34.42 \pm 17.45$ | $12.26 \pm 5.56$ | $0.00 \pm 0.00$ |
| BrainOmni | DEAP | $45.12 \pm 3.29$ | $22.33 \pm 2.66$ | $22.34 \pm 3.37$ | $-4.59 \pm 2.22$ |
| BrainWave | Dep-STAI | **$63.55 \pm 6.15$** | $51.34 \pm 4.83$ | **$43.19 \pm 3.58$** | **$14.97 \pm 5.32$** |
| NeuroGPT-E | Dep-STAI | $61.06 \pm 9.02$ | $55.27 \pm 7.26$ | $39.97 \pm 11.16$ | $12.89 \pm 15.50$ |
| LaBraM | Dep-STAI | $60.61 \pm 8.63$ | $36.32 \pm 10.35$ | $39.44 \pm 4.62$ | $8.08 \pm 9.25$ |
| BrainBERT | Dep-STAI | $60.47 \pm 5.70$ | $47.52 \pm 11.04$ | $36.72 \pm 8.81$ | $6.41 \pm 12.62$ |
| BrainOmni | Dep-STAI | $59.63 \pm 5.74$ | $52.77 \pm 5.06$ | $37.65 \pm 4.63$ | $10.51 \pm 6.16$ |
| BFM | Dep-STAI | $57.63 \pm 7.73$ | $45.30 \pm 8.61$ | $38.79 \pm 5.77$ | $9.85 \pm 11.66$ |
| NeuroGPT-D | Dep-STAI | $56.59 \pm 5.54$ | **$60.52 \pm 0.96$** | $25.13 \pm 0.25$ | $0.00 \pm 0.00$ |
| EEGPT | Dep-STAI | $56.47 \pm 5.68$ | $51.40 \pm 6.37$ | $36.14 \pm 5.89$ | $2.47 \pm 8.03$ |
| BIOT | Dep-STAI | $56.34 \pm 9.04$ | $36.91 \pm 7.44$ | $38.61 \pm 6.88$ | $7.31 \pm 8.81$ |
| Mbrain | Dep-STAI | $55.39 \pm 3.76$ | $60.34 \pm 2.61$ | $31.65 \pm 6.08$ | $5.37 \pm 5.93$ |
| REVE | Dep-STAI | $54.33 \pm 11.41$ | $54.42 \pm 10.31$ | $35.13 \pm 8.12$ | $7.86 \pm 20.02$ |
| CBraMod | Dep-STAI | $54.15 \pm 3.69$ | $60.38 \pm 1.66$ | $35.42 \pm 2.88$ | $4.92 \pm 5.24$ |
| Bendr | Dep-STAI | $53.78 \pm 3.28$ | $47.73 \pm 3.28$ | $36.40 \pm 3.39$ | $4.71 \pm 4.14$ |
| Brant1 | Dep-STAI | $53.49 \pm 2.97$ | **$60.52 \pm 0.96$** | $25.13 \pm 0.25$ | $0.00 \pm 0.00$ |
| SppEEGNet | Dep-STAI | $51.45 \pm 2.35$ | $46.38 \pm 2.85$ | $33.80 \pm 1.49$ | $0.72 \pm 2.43$ |
| NeuroLM | Dep-STAI | $50.32 \pm 4.22$ | $57.65 \pm 4.45$ | $26.81 \pm 3.51$ | $0.85 \pm 1.81$ |
| REVE | EEGMMIDB-I | **$82.10 \pm 1.50$** | **$58.83 \pm 2.30$** | **$59.53 \pm 1.91$** | **$45.11 \pm 3.06$** |
| NeuroGPT-E | EEGMMIDB-I | $77.66 \pm 1.07$ | $54.37 \pm 1.44$ | $54.71 \pm 1.33$ | $39.16 \pm 1.93$ |
| BrainOmni | EEGMMIDB-I | $75.65 \pm 1.17$ | $51.34 \pm 1.41$ | $51.69 \pm 1.28$ | $35.12 \pm 1.89$ |
| LaBraM | EEGMMIDB-I | $59.17 \pm 2.00$ | $31.16 \pm 1.81$ | $30.78 \pm 1.29$ | $7.83 \pm 3.18$ |
| CBraMod | EEGMMIDB-I | $58.41 \pm 1.67$ | $30.47 \pm 2.07$ | $31.72 \pm 1.80$ | $7.34 \pm 2.78$ |
| Bendr | EEGMMIDB-I | $55.02 \pm 0.65$ | $29.04 \pm 0.73$ | $28.97 \pm 0.78$ | $5.39 \pm 0.99$ |
| BFM | EEGMMIDB-I | $52.45 \pm 1.35$ | $26.72 \pm 0.70$ | $26.63 \pm 0.70$ | $2.29 \pm 0.93$ |
| EEGPT | EEGMMIDB-I | $51.85 \pm 0.75$ | $26.69 \pm 1.60$ | $27.42 \pm 0.89$ | $2.33 \pm 2.23$ |
| Mbrain | EEGMMIDB-I | $51.07 \pm 1.04$ | $25.65 \pm 0.94$ | $23.15 \pm 2.73$ | $0.94 \pm 1.11$ |
| NeuroLM | EEGMMIDB-I | $51.03 \pm 1.13$ | $25.30 \pm 0.63$ | $19.60 \pm 5.88$ | $0.50 \pm 0.81$ |
| BrainWave | EEGMMIDB-I | $50.99 \pm 0.51$ | $25.18 \pm 1.05$ | $25.13 \pm 2.51$ | $0.09 \pm 1.38$ |
| Brant1 | EEGMMIDB-I | $50.93 \pm 0.68$ | $24.92 \pm 0.37$ | $10.02 \pm 0.07$ | $0.00 \pm 0.00$ |
| SppEEGNet | EEGMMIDB-I | $50.91 \pm 0.99$ | $25.67 \pm 1.01$ | $25.46 \pm 1.19$ | $0.81 \pm 1.30$ |
| BIOT | EEGMMIDB-I | $50.05 \pm 0.59$ | $24.64 \pm 0.31$ | $24.70 \pm 0.58$ | $-0.55 \pm 0.44$ |
| BrainBERT | EEGMMIDB-I | $49.19 \pm 0.44$ | $24.91 \pm 0.35$ | $23.71 \pm 1.10$ | $-0.25 \pm 0.47$ |
| NeuroGPT-D | EEGMMIDB-I | $49.12 \pm 1.39$ | $24.57 \pm 0.67$ | $20.59 \pm 5.16$ | $-0.68 \pm 0.76$ |
| REVE | EEGMMIDB-R | **$82.39 \pm 0.50$** | **$58.60 \pm 1.07$** | **$58.95 \pm 0.90$** | **$44.80 \pm 1.42$** |
| NeuroGPT-E | EEGMMIDB-R | $77.48 \pm 1.72$ | $53.55 \pm 2.27$ | $54.12 \pm 1.89$ | $38.07 \pm 3.02$ |
| BrainOmni | EEGMMIDB-R | $74.32 \pm 0.99$ | $49.36 \pm 1.00$ | $49.69 \pm 1.72$ | $32.47 \pm 1.35$ |
| CBraMod | EEGMMIDB-R | $57.50 \pm 1.11$ | $29.54 \pm 1.66$ | $30.75 \pm 1.17$ | $6.06 \pm 2.20$ |
| Bendr | EEGMMIDB-R | $55.15 \pm 0.58$ | $29.48 \pm 1.01$ | $29.42 \pm 1.02$ | $5.95 \pm 1.35$ |
| LaBraM | EEGMMIDB-R | $54.67 \pm 1.85$ | $28.09 \pm 1.91$ | $27.98 \pm 2.47$ | $4.12 \pm 2.70$ |
| BFM | EEGMMIDB-R | $52.50 \pm 0.69$ | $26.82 \pm 0.56$ | $26.81 \pm 0.61$ | $2.42 \pm 0.75$ |
| SppEEGNet | EEGMMIDB-R | $51.75 \pm 0.42$ | $26.07 \pm 0.56$ | $25.78 \pm 0.75$ | $1.39 \pm 0.76$ |
| Mbrain | EEGMMIDB-R | $51.58 \pm 1.01$ | $25.60 \pm 1.20$ | $23.44 \pm 5.51$ | $0.82 \pm 1.57$ |
| NeuroLM | EEGMMIDB-R | $51.52 \pm 2.16$ | $25.61 \pm 1.37$ | $21.63 \pm 7.19$ | $0.90 \pm 1.73$ |
| EEGPT | EEGMMIDB-R | $51.11 \pm 0.76$ | $25.47 \pm 1.30$ | $25.06 \pm 2.76$ | $0.62 \pm 1.73$ |
| NeuroGPT-D | EEGMMIDB-R | $50.46 \pm 0.12$ | $25.18 \pm 0.20$ | $23.53 \pm 1.98$ | $0.28 \pm 0.31$ |
| Brant1 | EEGMMIDB-R | $50.33 \pm 0.83$ | $25.05 \pm 0.15$ | $10.03 \pm 0.04$ | $0.00 \pm 0.00$ |
| BIOT | EEGMMIDB-R | $50.13 \pm 0.83$ | $25.19 \pm 0.82$ | $25.19 \pm 1.21$ | $0.27 \pm 1.11$ |
| BrainWave | EEGMMIDB-R | $50.00 \pm 0.55$ | $24.82 \pm 0.86$ | $25.74 \pm 1.53$ | $-0.36 \pm 1.08$ |
| BrainBERT | EEGMMIDB-R | $50.00 \pm 0.54$ | $25.12 \pm 0.43$ | $24.59 \pm 1.14$ | $0.07 \pm 0.56$ |
| REVE | ISRUC | **$91.55 \pm 1.22$** | **$69.02 \pm 2.96$** | **$67.95 \pm 3.30$** | **$59.78 \pm 3.59$** |
| BrainOmni | ISRUC | $91.18 \pm 1.88$ | $67.64 \pm 2.68$ | $65.78 \pm 4.18$ | $58.09 \pm 2.97$ |
| BrainWave | ISRUC | $88.55 \pm 0.87$ | $62.33 \pm 5.47$ | $60.63 \pm 3.53$ | $52.24 \pm 6.37$ |
| EEGPT | ISRUC | $87.31 \pm 2.16$ | $60.04 \pm 4.96$ | $56.12 \pm 8.26$ | $48.61 \pm 7.03$ |
| Mbrain | ISRUC | $86.78 \pm 4.02$ | $63.06 \pm 6.65$ | $58.67 \pm 7.55$ | $51.43 \pm 8.41$ |
| CBraMod | ISRUC | $85.68 \pm 4.54$ | $54.17 \pm 8.24$ | $53.31 \pm 7.39$ | $41.87 \pm 9.37$ |
| LaBraM | ISRUC | $85.13 \pm 2.32$ | $61.24 \pm 2.09$ | $58.67 \pm 4.10$ | $50.25 \pm 2.78$ |
| Brant1 | ISRUC | $83.81 \pm 1.76$ | $54.78 \pm 1.18$ | $51.66 \pm 3.94$ | $42.70 \pm 1.63$ |
| NeuroGPT-E | ISRUC | $80.41 \pm 2.07$ | $43.91 \pm 10.66$ | $47.31 \pm 6.21$ | $31.89 \pm 10.07$ |
| BrainBERT | ISRUC | $80.37 \pm 2.56$ | $46.81 \pm 5.11$ | $48.72 \pm 4.51$ | $33.21 \pm 5.81$ |
| BIOT | ISRUC | $77.76 \pm 4.36$ | $45.93 \pm 6.76$ | $46.55 \pm 2.91$ | $32.22 \pm 7.13$ |
| BFM | ISRUC | $77.60 \pm 2.83$ | $47.49 \pm 4.11$ | $45.80 \pm 2.73$ | $26.41 \pm 14.81$ |
| Bendr | ISRUC | $72.22 \pm 3.91$ | $42.71 \pm 6.61$ | $39.59 \pm 4.38$ | $26.87 \pm 7.62$ |
| NeuroLM | ISRUC | $66.92 \pm 4.76$ | $27.27 \pm 11.11$ | $18.29 \pm 8.41$ | $7.35 \pm 7.35$ |
| NeuroGPT-D | ISRUC | $60.75 \pm 7.33$ | $30.70 \pm 7.46$ | $18.16 \pm 8.98$ | $5.98 \pm 10.23$ |
| SppEEGNet | ISRUC | $52.79 \pm 1.16$ | $16.55 \pm 3.30$ | $23.49 \pm 2.17$ | $1.24 \pm 0.87$ |
| NeuroGPT-E | SEED-IV | **$56.65 \pm 1.10$** | **$29.88 \pm 0.56$** | **$30.89 \pm 1.27$** | $0.27 \pm 0.52$ |
| REVE | SEED-IV | $53.95 \pm 1.03$ | $28.09 \pm 1.53$ | $28.03 \pm 1.56$ | **$3.93 \pm 1.91$** |
| BrainOmni | SEED-IV | $52.93 \pm 1.49$ | $27.71 \pm 1.52$ | $27.69 \pm 1.43$ | $3.47 \pm 1.85$ |
| BrainWave | SEED-IV | $52.73 \pm 2.33$ | $27.44 \pm 1.77$ | $27.24 \pm 1.71$ | $2.26 \pm 2.63$ |
| Mbrain | SEED-IV | $52.14 \pm 1.46$ | $26.49 \pm 1.49$ | $25.46 \pm 2.11$ | $1.38 \pm 1.97$ |
| BFM | SEED-IV | $51.78 \pm 1.06$ | $26.62 \pm 0.70$ | $25.93 \pm 0.22$ | $1.21 \pm 0.38$ |
| CBraMod | SEED-IV | $51.13 \pm 0.98$ | $26.13 \pm 0.94$ | $25.56 \pm 1.45$ | $1.01 \pm 1.01$ |
| BrainBERT | SEED-IV | $51.05 \pm 2.66$ | $24.98 \pm 1.59$ | $24.47 \pm 0.89$ | $0.28 \pm 1.19$ |
| EEGPT | SEED-IV | $50.72 \pm 1.21$ | $25.16 \pm 1.03$ | $25.42 \pm 1.19$ | $0.58 \pm 1.61$ |
| LaBraM | SEED-IV | $50.67 \pm 1.06$ | $25.30 \pm 1.41$ | $25.60 \pm 0.95$ | $0.77 \pm 1.39$ |
| NeuroGPT-D | SEED-IV | $50.32 \pm 0.19$ | $26.63 \pm 0.01$ | $19.51 \pm 8.04$ | $0.01 \pm 0.02$ |
| NeuroLM | SEED-IV | $50.25 \pm 0.50$ | $25.06 \pm 0.49$ | $18.27 \pm 4.89$ | $0.13 \pm 0.17$ |
| Brant1 | SEED-IV | $50.24 \pm 1.11$ | $25.58 \pm 0.93$ | $10.18 \pm 0.29$ | $0.00 \pm 0.00$ |
| Bendr | SEED-IV | $50.03 \pm 0.45$ | $24.90 \pm 0.39$ | $24.96 \pm 0.40$ | $-0.06 \pm 0.55$ |
| SppEEGNet | SEED-IV | $49.87 \pm 0.22$ | $24.63 \pm 0.25$ | $24.90 \pm 0.14$ | $-0.09 \pm 0.21$ |
| BIOT | SEED-IV | $48.34 \pm 0.38$ | $23.84 \pm 0.86$ | $23.62 \pm 0.37$ | $-1.83 \pm 0.32$ |
| Mbrain | SleepEDF | **$94.31 \pm 1.44$** | **$78.95 \pm 4.36$** | **$73.36 \pm 4.63$** | **$70.89 \pm 5.90$** |
| NeuroGPT-E | SleepEDF | $94.10 \pm 1.14$ | $78.26 \pm 3.27$ | $71.40 \pm 2.54$ | $69.58 \pm 4.47$ |
| Brant1 | SleepEDF | $93.77 \pm 1.47$ | $72.65 \pm 7.23$ | $71.89 \pm 3.92$ | $63.86 \pm 8.34$ |
| CBraMod | SleepEDF | $93.41 \pm 1.42$ | $73.44 \pm 4.23$ | $70.06 \pm 3.37$ | $64.22 \pm 5.37$ |
| LaBraM | SleepEDF | $92.80 \pm 1.04$ | $75.69 \pm 3.87$ | $69.75 \pm 3.13$ | $66.92 \pm 4.97$ |
| BrainBERT | SleepEDF | $91.38 \pm 2.09$ | $68.21 \pm 7.18$ | $66.60 \pm 4.08$ | $57.83 \pm 7.89$ |
| Bendr | SleepEDF | $90.93 \pm 1.34$ | $69.31 \pm 5.64$ | $65.48 \pm 2.57$ | $59.09 \pm 6.25$ |
| EEGPT | SleepEDF | $90.92 \pm 0.37$ | $68.10 \pm 2.79$ | $66.82 \pm 0.72$ | $57.83 \pm 2.59$ |
| BrainWave | SleepEDF | $89.14 \pm 1.87$ | $66.78 \pm 5.43$ | $58.53 \pm 5.92$ | $51.56 \pm 10.36$ |
| BFM | SleepEDF | $88.52 \pm 1.50$ | $64.54 \pm 3.72$ | $60.42 \pm 2.20$ | $55.77 \pm 5.27$ |
| BrainOmni | SleepEDF | $87.43 \pm 0.42$ | $62.70 \pm 3.12$ | $61.80 \pm 0.30$ | $50.63 \pm 2.49$ |
| REVE | SleepEDF | $78.17 \pm 37.76$ | $63.23 \pm 29.55$ | $71.75 \pm 4.09$ | $56.11 \pm 25.66$ |
| NeuroGPT-D | SleepEDF | $64.60 \pm 3.26$ | $48.05 \pm 3.56$ | $27.09 \pm 3.00$ | $9.95 \pm 6.81$ |
| NeuroLM | SleepEDF | $60.57 \pm 5.65$ | $17.60 \pm 3.78$ | $15.24 \pm 6.79$ | $18.90 \pm 26.27$ |
| BIOT | SleepEDF | $57.77 \pm 16.01$ | $34.74 \pm 16.40$ | $24.36 \pm 13.59$ | $6.98 \pm 14.48$ |
| SppEEGNet | SleepEDF | $56.89 \pm 0.83$ | $21.51 \pm 1.34$ | $25.56 \pm 1.04$ | $10.37 \pm 2.80$ |