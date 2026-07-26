import torch
from torch import nn

'''
    Brant: The implementation is not fully released.
        Pretrained model weights are available on Hugging Face.
        Access to the complete code may be obtained by contacting the authors.

    BrainWave: The implementation is currently unavailable. 
        The authors state that the code will be released upon paper acceptance.
'''

from utils.get_data import default_get_data, default_get_data_with_pos
# from datasets.Brant_dataset import Brant_Dataset
# from datasets.BrainWave_dataset import BrainWave_Dataset
from datasets.default_dataset import DefaultDataset
from datasets.NeuroLM_dataset import NeuroLMDataset
from datasets.BFM_dataset import BFMDataset
from datasets.EEGPT_dataset import EEGPTDataset
from datasets.NeuroGPT_dataset import NeuroGPTDataset
from datasets.BrainOmni_dataset import BrainOminiDataset
from datasets.REVE_dataset import REVEDataset
from model.BIOT.BIOT import BIOT_Trainer, BIOT
from model.BrainBERT.BrainBERT import BrainBERT_Trainer, BrainBERT
# from model.Brant.Brant import Brant, Brant_Trainer  
# from model.BrainWave.BrainWave import BrainWave_Trainer, BrainWave
from model.LaBraM.LaBraM import LaBraM, LaBraM_Trainer
from model.Bendr.Bendr import BENDR, Bendr_Trainer
from model.SppEEGNet.SppEEGNet import SppEEGNet, SppEEGNet_Trainer
from model.Mbrain.Mbrain import Mbrain, Mbrain_Trainer
from model.BrainOmni.BrainOmni import BrainOmni_Main, BrainOmni_Trainer
from model.BFM.BFM import BFM_Trainer, BFM
from model.CBraMod.CBraMod import CBraMod, CBraMod_Trainer
from model.NeuroGPT.NeuroGPT import NeuroGPT, NeuroGPT_Trainer
from model.NeuroLM.NeuroLM import NeuroLM, NeuroLM_Trainer
from model.EEGPT.EEGPT import EEGPT, EEGPT_Trainer
from model.REVE.REVE import REVE, REVE_Trainer
from model.MVPFormer.MVPFormer import MVPFormer, MVPFormer_Trainer
from model.CodeBrain.CodeBrain import CodeBrain, CodeBrain_Trainer

from utils.metrics import BinaryClassMetrics, MultiClassMetrics

### To add a new dataset, please update the following dicts

get_data_dict = {
    'MAYO' : default_get_data,
    'FNUSA' : default_get_data,
    'CHBMIT' : default_get_data,
    'Siena' : default_get_data,
    'SleepEDFx' : default_get_data,
    'RepOD': default_get_data,
    'UCSD_ON': default_get_data,
    'UCSD_OFF': default_get_data,
    'ISRUC': default_get_data,
    'ADFD': default_get_data,
    'ADHD_Adult': default_get_data_with_pos,
    'ADHD_Child': default_get_data,
    'Depression_122_BDI': default_get_data,
    'Depression_122_STAI': default_get_data,
    'Schizophrenia_28': default_get_data,
    'MPHCE_mdd': default_get_data,
    'MPHCE_state': default_get_data,
    'SEED_IV': default_get_data,
    'SD_71': default_get_data,
    'EEGMat': default_get_data,
    'DEAP': default_get_data,
    'EEGMMIDB_R': default_get_data,
    'EEGMMIDB_I': default_get_data,
    'AD-65': default_get_data,
    'BCI-2a': default_get_data,
    'Chisco_read': default_get_data,
    'Chisco_imagine': default_get_data,
}

metrics_dict = {
    'MAYO': BinaryClassMetrics,
    'FNUSA': BinaryClassMetrics,
    'CHBMIT': BinaryClassMetrics,
    'Siena': BinaryClassMetrics,
    'SleepEDFx': MultiClassMetrics,
    'RepOD': BinaryClassMetrics,
    'UCSD_ON': BinaryClassMetrics,
    'UCSD_OFF': BinaryClassMetrics,
    'ISRUC': MultiClassMetrics,
    'ADFD': BinaryClassMetrics,
    'ADHD_Adult': BinaryClassMetrics,
    'ADHD_Child': BinaryClassMetrics,
    'Depression_122_BDI': BinaryClassMetrics,
    'Schizophrenia_28': BinaryClassMetrics,
    'MPHCE_mdd': BinaryClassMetrics,
    'MPHCE_state': MultiClassMetrics,
    'SEED_IV': MultiClassMetrics,
    'SD_71': BinaryClassMetrics,
    'EEGMat': BinaryClassMetrics,
    'DEAP': MultiClassMetrics,
    'EEGMMIDB_R': MultiClassMetrics,
    'EEGMMIDB_I': MultiClassMetrics,
    'Depression_122_STAI': MultiClassMetrics,
    'AD-65': BinaryClassMetrics,
    'BCI-2a': MultiClassMetrics,
    'Chisco_read': MultiClassMetrics,
    'Chisco_imagine': MultiClassMetrics,
}

### To add a new method, please update the following dicts

dataset_class_dict = {
    'BrainBERT': DefaultDataset,
    # 'Brant1': Brant_Dataset,
    # 'BrainWave': BrainWave_Dataset,
    'LaBraM': DefaultDataset,
    'BIOT': DefaultDataset,
    'Bendr': DefaultDataset,
    'SppEEGNet': DefaultDataset,
    'Mbrain': DefaultDataset,
    'CRT': DefaultDataset,
    'BFM': BFMDataset,
    'CBraMod': DefaultDataset,
    'NeuroGPT': NeuroGPTDataset,
    'NeuroLM': NeuroLMDataset,
    'EEGPT': EEGPTDataset,
    'EEGNet': DefaultDataset,
    'SPaRCNet': DefaultDataset,
    'DeprNet': DefaultDataset,
    'CCNSE': DefaultDataset,
    'BrainOmni': BrainOminiDataset,
    'REVE': REVEDataset,
    'MVPFormer': MVPFormer_Trainer,
    'CodeBrain': CodeBrain_Trainer,
}

trainer_dict = {
    'BrainBERT': BrainBERT_Trainer,
    # 'Brant1': Brant_Trainer,
    # 'BrainWave': BrainWave_Trainer,
    'LaBraM': LaBraM_Trainer,
    'BIOT': BIOT_Trainer,
    'Bendr': Bendr_Trainer,
    'SppEEGNet': SppEEGNet_Trainer,
    'Mbrain': Mbrain_Trainer,
    'BFM': BFM_Trainer,
    'CBraMod': CBraMod_Trainer,
    'NeuroGPT': NeuroGPT_Trainer,
    'NeuroLM': NeuroLM_Trainer,
    'EEGPT': EEGPT_Trainer,
    'BrainOmni': BrainOmni_Trainer,
    'REVE': REVE_Trainer,
    'MVPFormer': MVPFormer_Trainer,
    'CodeBrain': CodeBrain_Trainer,
}

model_dict = {
    'BrainBERT': BrainBERT,
    # 'Brant': Brant,
    # 'BrainWave': BrainWave,
    'LaBraM': LaBraM,
    'BIOT': BIOT,
    'Bendr': BENDR,
    'SppEEGNet': SppEEGNet,
    'Mbrain': Mbrain,
    'BFM': BFM,
    'CBraMod': CBraMod,
    'NeuroGPT': NeuroGPT,
    'NeuroLM': NeuroLM,
    'EEGPT': EEGPT,
    'BrainOmni': BrainOmni_Main,
    'REVE': REVE,
    'MVPFormer': MVPFormer,
    'CodeBrain': CodeBrain,
}


