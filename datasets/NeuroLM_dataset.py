import tiktoken
import torch
import numpy as np
import os
import json
from scipy.signal import resample
from torch.utils.data import Dataset, DataLoader

from data_preprocess.utils import _std_data_segment


standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", \
    'pad', 'I1', 'I2', \
    'T7-FT9', 'FT10-T8', 'O2-A1', 'F4-A1', 'FZ-CZ', 'FT9-FT10', 'C4-A1', 'FPZ-CZ', 'CZ-PZ', 'C3-A2', 'P7-T7', 'F3-A2', 'O1-A2'    # add new channels
]


class NeuroLMDataset(Dataset):
    def __init__(self, args, x, y, is_train=True, is_instruct=True):
        """
        Initialize the EEG dataset for NeuroLM model.
        
        Args:
            args: Configuration arguments containing:
                - patch_len: Length of each EEG patch/window
                - n_process_loader: Number of processes for data loading
                - gpt_path: Path to GPT tokenizer
                - downstream: Task type ('disorder' or emotion classification)
            x: EEG data with shape (seq_num, ch_num, N)
            y: Labels with shape (seq_num, )
        """
        pos = None
        if isinstance(x, dict):
            pos = x.get("pos", None)
            x = x['x']
        self.seq_num, self.ch_num, N = x.shape
        x = x[..., :args.seq_len*args.patch_len]
        self.raw_x = x
        x = self.safe_resample(x, args.patch_len)
        # args.patch_len = 200
        x = _std_data_segment(x)  # Apply time-level normalization
        
        self.x = x
        self.y = y
        self.pos = pos
        self.window_size = 200
        self.eeg_max_len = 276
        self.text_max_len = 32
        self.mode = args.run_mode
        self.is_instruct = is_instruct
        self.is_val = False if args.run_mode == 'finetune' and is_train else True
        self.ch_name = None
        
        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)
        
        channel_path = os.path.join(args.full_data_path, 'channels_lst.json')
        if os.path.exists(channel_path):
            with open(channel_path, 'r') as f:
                self.ch_name = json.load(f)
           
         # initialize GPT tokenizer
        enc = tiktoken.get_encoding('gpt2')
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        
        # Create prompt based on downstream task
        if self.is_instruct:
            if args.downstream == 'disorder':
                self.prompt = torch.IntTensor([50257] + encode('Question: Does the patient have a disease? Answer:'))
                if args.dataset == 'SleepEDFx' or args.dataset == 'ISRUC':
                    self.text = {
                        0: torch.IntTensor([50257] + encode('Question: Which sleep stage does this signal belong to? Answer: W <|endoftext|>')),
                        1: torch.IntTensor([50257] + encode('Question: Which sleep stage does this signal belong to? Answer: N1 <|endoftext|>')),
                        2: torch.IntTensor([50257] + encode('Question: Which sleep stage does this signal belong to? Answer: N2 <|endoftext|>')),
                        3: torch.IntTensor([50257] + encode('Question: Which sleep stage does this signal belong to? Answer: N3 <|endoftext|>')),
                        4: torch.IntTensor([50257] + encode('Question: Which sleep stage does this signal belong to? Answer: R <|endoftext|>')),
                    }
                else:
                    self.text = {
                        0: torch.IntTensor([50257] + encode('Question: Does the patient have a disease? Answer: health <|endoftext|>')),
                        1: torch.IntTensor([50257] + encode('Question: Does the patient have a disease? Answer: illness <|endoftext|>')),
                        2: torch.IntTensor([50257] + encode('Question: Does the patient have a disease? Answer: serious <|endoftext|>'))    # different kind  
                    }
            elif args.dataset == 'EEGMat':
                self.prompt = torch.IntTensor([50257] + encode(f'Question: Is the patient performing a mental arithmetic task? Answer:'))
                self.text = {
                    0: torch.IntTensor([50257] + encode('Question: Is the patient performing a mental arithmetic task? Answer: No <|endoftext|>')),
                    1: torch.IntTensor([50257] + encode('Question: Is the patient performing a mental arithmetic task? Answer: Yes <|endoftext|>')),
                }
            elif args.downstream == 'concept':
                self.prompt = torch.IntTensor([50257] + encode(f'Question: Which concept type does the patient think of? Answer:'))
                if args.dataset == 'concept_old':
                    self.text = {
                        0: torch.IntTensor([50257] + encode('Question: Which concept type does the patient think of? Answer: toilet <|endoftext|>')),
                        1: torch.IntTensor([50257] + encode('Question: Which concept type does the patient think of? Answer: eat <|endoftext|>')),
                        2: torch.IntTensor([50257] + encode('Question: Which concept type does the patient think of? Answer: sleep <|endoftext|>')),
                        3: torch.IntTensor([50257] + encode('Question: Which concept type does the patient think of? Answer: cellphone <|endoftext|>')),
                        4: torch.IntTensor([50257] + encode('Question: Which concept type does the patient think of? Answer: healthcare <|endoftext|>')),
                    }
                else:
                    self.text = {}
                    with open(args.label_path, 'r') as f:
                        text_data = json.load(f).values()
                    for i, text in enumerate(text_data):
                        self.text[i] = torch.IntTensor([50257] + encode(f'Question: Which concept type does the patient think of? Answer: {text} <|endoftext|>'))
                    
            else:
                # For emotion/ MI classification (modify labels as needed)
                self.prompt = torch.IntTensor([50257] + encode(f'Question: Which {args.downstream} type does this EEG segment belong to? Answer:'))
                if args.dataset == 'SEED_IV':
                    self.text = {
                        0: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Happy <|endoftext|>')),
                        1: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Neutral <|endoftext|>')),
                        2: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Sad <|endoftext|>')),
                        3: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Fear <|endoftext|>'))
                    }
                elif args.dataset == 'DEAP':
                    self.text = {
                        0: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Low Valence and Low Dominance <|endoftext|>')),
                        1: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: Low Valence and High Dominance <|endoftext|>')),
                        2: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: High Valence and Low Dominance <|endoftext|>')),
                        3: torch.IntTensor([50257] + encode('Question: Which emotion type does this EEG segment belong to? Answer: High Valence and High Dominance <|endoftext|>'))
                    }
                elif 'EEGMMIDB' in args.dataset or 'BCI-2a' in args.dataset:
                    self.text = {
                        0: torch.IntTensor([50257] + encode(f'Question: Which {args.downstream} type does this EEG segment belong to? Answer: Single left fist <|endoftext|>')),
                        1: torch.IntTensor([50257] + encode(f'Question: Which {args.downstream} type does this EEG segment belong to? Answer: Single right fist <|endoftext|>')),
                        2: torch.IntTensor([50257] + encode(f'Question: Which {args.downstream} type does this EEG segment belong to? Answer: Both fists <|endoftext|>')),
                        3: torch.IntTensor([50257] + encode(f'Question: Which {args.downstream} type does this EEG segment belong to? Answer: Both feet <|endoftext|>'))
                    }
                    if 'BCI-2a' in args.dataset:
                        self.text[2] = torch.IntTensor([50257] + encode(f'Question: Which {args.downstream} type does this EEG segment belong to? Answer: Both feet <|endoftext|>'))
                        self.text[3] = torch.IntTensor([50257] + encode(f'Question: Which {args.downstream} type does this EEG segment belong to? Answer: Tongue movement <|endoftext|>'))
            args.prompt_len = len(self.prompt)


    def __getitem__(self, index):
        """
        Returns:
            X_eeg: Padded EEG data (eeg_max_len, window_size)
            text: Prompt tokens
            label: Ground truth label
            input_chans: Channel indices
            input_time: Time indices
            eeg_mask: Mask for EEG padding
            gpt_mask: Attention mask for GPT
        """
        eeg = self.x[index]
        label = self.y[index]
        
        seq_len = eeg.shape[1] // self.window_size
        eeg = torch.from_numpy(eeg.reshape(-1, self.window_size)).float()   # c (s w) -> (s c) w
        if self.pos is None:
            input_chans = self.get_chans(self.ch_name, seq_len)
        else:
            pos_i = self.pos[index]
            ch_name = np.array(self.ch_name)[pos_i]
            input_chans = self.get_chans(ch_name, seq_len)
        input_time = [i for i in range(seq_len) for _ in range(self.ch_num)]
        
        # without text
        if not self.is_instruct:
            input_chans = torch.IntTensor(input_chans)
            input_time = torch.IntTensor(input_time)
            gpt_mask = torch.tril(torch.ones(eeg.size(0), eeg.size(0))).view(1, eeg.size(0), eeg.size(0))
            for i in range(seq_len):
                gpt_mask[:, i * self.ch_num:(i + 1) * self.ch_num,  i * self.ch_num:(i + 1) * self.ch_num] = 1

            return eeg, self.y[index], input_chans, input_time
        
        # get X_text
        if self.is_val:
            text = self.prompt
        else:
            text = self.text[int(label)]
            # pad text to text_max_len
            valid_text_len = text.size(0)
            if self.text_max_len > valid_text_len:
                text_pad = torch.full((self.text_max_len,), fill_value=50256)
                text_pad[:valid_text_len] = text
                text = text_pad
        
        # pad eeg to eeg_max_len
        valid_eeg_len = eeg.shape[0]
        if self.eeg_max_len > valid_eeg_len:
            X_eeg = torch.zeros((self.eeg_max_len, self.window_size))
            X_eeg[:valid_eeg_len] = eeg
            eeg_mask = torch.ones(self.eeg_max_len)
            eeg_mask[valid_eeg_len:] = 0
            
            input_chans.extend([136] * (self.eeg_max_len - valid_eeg_len))  # 'pad'
            input_time.extend([0] * (self.eeg_max_len - valid_eeg_len))
        else:
            X_eeg = eeg
            eeg_mask = torch.ones(valid_eeg_len)
        
        input_chans = torch.IntTensor(input_chans)
        input_time = torch.IntTensor(input_time)
        
        # create GPT attention mask
        num_tokens = X_eeg.size(0) + text.size(0)
        gpt_mask = torch.tril(torch.ones(num_tokens, num_tokens)).view(1, num_tokens, num_tokens)
        # Ensure that EEG channels within the same time step can pay attention to each other
        for i in range(seq_len):
            gpt_mask[:, i * self.ch_num:(i + 1) * self.ch_num,  i * self.ch_num:(i + 1) * self.ch_num] = 1
        gpt_mask[:, :, valid_eeg_len:X_eeg.size(0)] = 0
        
        if self.is_val:
            return X_eeg, text, label, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), self.raw_x[index]
        
        Y_text = torch.full_like(text, fill_value=-1)
        prompt_len = self.prompt.size(0)
        Y_text[prompt_len - 1:valid_text_len - 1] = text[prompt_len:valid_text_len]
        
        return X_eeg, text, label, Y_text, input_chans, input_time, eeg_mask.bool(), gpt_mask.bool(), self.raw_x[index]


    def __len__(self):
        return self.seq_num

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers, drop_last=False, pin_memory=True,
                          shuffle=shuffle)
        
    def safe_resample(self, data, original_sfreq, target_sfreq=200):
        """        
        Param:
            data: NumPy,  (n_channels, n_times)
            original_sfreq: origin sampling (Hz)
            target_sfreq: target sampling (default 200Hz)
        
        return:
            resampled_data: (n_channels, new_n_times)
        """
        seq_len, ch_num, N = data.shape
        data = data.reshape(-1, N)
        n_samples = int(data.shape[1] * target_sfreq / original_sfreq)
        resampled = resample(data, n_samples, axis=1)
        resampled = resampled.reshape(seq_len, ch_num, -1)
        return resampled


    def get_chans(self, ch_names, seq_len):
        if ch_names is None:
            chans = list(range(self.ch_num))
            
        else:
            chans = []
            for ch_name in ch_names:
                chans.append(standard_1020.index(ch_name.upper()))
        return chans * seq_len