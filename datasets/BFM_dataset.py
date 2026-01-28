import numpy as np
import torch
import os
import pandas as pd
import json
from torch.utils.data import Dataset, DataLoader
from gluonts.transform import (
    InstanceSplitter,
    TestSplitSampler,
    ValidationSplitSampler,
    ExpectedNumInstanceSampler,
)
from chronos import ChronosConfig

from data_preprocess.utils import _std_data_segment

class BFMDataset(Dataset):
    def __init__(self, args, x, y,
                context_length: int = 512,
                prediction_length: int = 64,
                min_past: int = 64,
                is_train: bool = True,
                model_type: str = "seq2seq",
                seed: int = 0,
                drop_prob: float = 0.0,
                random_seed: int = 42,
                max_channels_per_seq: int = 10,
                ):
        assert model_type in ["seq2seq", "causal"]
        # x: (seq_num, ch_num, N)
        # y: (seq_num, )
        if isinstance(x, dict):
            x = x['x']
        self.seq_num, self.ch_num, N = x.shape
        x = _std_data_segment(x)    # time level normalization

        # self.ch_id = [ch for _ in range(self.seq_num) for ch in range(self.ch_num)]
        # self.x = x.reshape(-1, N)
        # self.y = np.repeat(y, self.ch_num)
        
        # random choose
        self.rng = np.random.RandomState(random_seed)
        sequence_seeds = self.rng.randint(0, 1000, size=self.seq_num)
        self.max_channels_per_seq = min(max_channels_per_seq, self.ch_num)
        self.x = []
        self.y = []
        self.ch_id = []
        
        for seq_idx in range(self.seq_num):
            if self.ch_num > self.max_channels_per_seq:
                seq_rng = np.random.RandomState(sequence_seeds[seq_idx])
                selected_chs = seq_rng.choice(
                    self.ch_num, 
                    size=self.max_channels_per_seq, 
                    replace=False
                )
            else:
                selected_chs = np.arange(self.ch_num)
            
            for ch_idx in selected_chs:
                self.x.append(x[seq_idx, ch_idx, :])
                self.y.append(y[seq_idx])
                self.ch_id.append(ch_idx)
        
        self.is_train = is_train 
        self.mode = 'training' if is_train and args.run_mode == 'finetune' else 'validation'
        if args.run_mode != 'finetune':
            self.mode = 'test'
            self.is_train = False
        
        chronos_config = ChronosConfig(
            tokenizer_class="MeanScaleUniformBins",
            tokenizer_kwargs={'low_limit': -15.0, 'high_limit': 15.0},
            n_tokens=args.vocab_size,
            n_special_tokens=2,
            pad_token_id=args.pad_token_id,
            eos_token_id=args.eos_token_id,
            use_eos_token=True,
            model_type=model_type,
            context_length=context_length,
            prediction_length=prediction_length,
            num_samples=20,
            temperature=1.0,
            top_k=50,
            top_p=1.0,
        )
        tokenizer = chronos_config.create_tokenizer()
        self.tokenizer = tokenizer
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.model_type = model_type
        self.drop_prob = drop_prob if model_type == "seq2seq" else 0.0
        self.seed = seed
        self.min_past = min_past or prediction_length

        self.nProcessLoader = args.n_process_loader
        self.reload_pool = torch.multiprocessing.Pool(self.nProcessLoader)
        self.rng = np.random.RandomState(seed)
        
        # Create instance splitter based on mode
        self.instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
            "test": TestSplitSampler(),
        }[self.mode]
        
        self.instance_splitter = InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=self.instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )
        
        assert x.shape[-1] >= context_length + prediction_length, \
            f"Sequence length {x.shape[-1]} < required {context_length + prediction_length}"


    def __getitem__(self, index):
        entry = {
            "start": 0,  # virtual timestamp
            "target": self.x[index],
            "feat_static_cat": [0],
            "channel_id": str(self.ch_id[index]),
        }
        
        if self.mode == "training" and self.drop_prob > 0:
            mask = self.rng.random(len(entry["target"])) < self.drop_prob
            entry["target"][mask] = np.nan
        
        transformed_data = self.instance_splitter.apply([entry], is_train=self.is_train)
        split_entry = list(transformed_data)[0]

        # tokenizer
        past_target = torch.tensor(split_entry["past_target"]).unsqueeze(0)     # (1, context_length)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(past_target)
        
        future_target = torch.tensor(split_entry["future_target"]).unsqueeze(0)  # (1, prediction_length)
        if future_target.numel() == 0:
            # generate virtual labels and mask
            dummy_length = self.prediction_length
            labels = torch.full((1, dummy_length), -100)
            labels_mask = torch.zeros((1, dummy_length))
            
        else:
            labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
            labels[labels_mask == 0] = -100     # mask padding

        if self.model_type == "causal":
            pad_start_idx = np.searchsorted(1 - split_entry["past_is_pad"], 1)
            padded_input_ids, obs_input_ids = torch.tensor_split(input_ids, [pad_start_idx], dim=-1)
            padded_attention_mask, obs_attention_mask = torch.tensor_split(attention_mask, [pad_start_idx], dim=-1)
            
            input_ids = torch.cat([obs_input_ids, labels, padded_input_ids], axis=-1)
            attention_mask = torch.cat([obs_attention_mask, labels_mask, padded_attention_mask], axis=-1)
            labels = input_ids.clone()
            input_ids[~attention_mask] = self.tokenizer.config.pad_token_id
            labels[~attention_mask] = -100
        
        
        input_ids = torch.squeeze(input_ids)                  # (context_length)
        attention_mask = torch.squeeze(attention_mask)        # (context_length)
        labels = torch.squeeze(labels)                        # (prediction_length)
        original_labels =  torch.tensor(self.y[index])        # scalar
        
        return original_labels, input_ids, attention_mask, labels


    def __len__(self):
        return len(self.x)

    def get_data_loader(self, batch_size, shuffle=False, num_workers=0):
        return DataLoader(self,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          drop_last=False, pin_memory=True,
                          shuffle=shuffle,)