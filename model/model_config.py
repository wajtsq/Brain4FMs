from dataclasses import dataclass

'''
    Please download the model weights from the official website and set `{model}_path` accordingly.
    MBrain: Pretrained model weights are not publicly available.
        Access requires contacting the authors of the original paper.

    BrainWave: The implementation is currently unavailable. 
        The authors state that the code will be released upon paper acceptance.
'''

@dataclass
class ModelPathArgs:
    BrainBERT_path: str = '/pretrained_weights/BrainBERT/stft_large_pretrained.pth'
    # Brant_root_path: str = '/pretrained_weights/Brant/'
    # BrainWave_path: str = '/pretrained_weights/BrainWave/'
    LaBraM_path: str = '/pretrained_weights/LaBraM/labram-base.pth'
    LaBraM_vq_path: str = '/pretrained_weights/LaBraM/vqnsp.pth'
    BIOT_path: str = '/pretrained_weights/BIOT/EEG-six-datasets-18-channels.ckpt'
    Bendr_path: str = '/pretrained_weights/Bendr/contextualizer.pt'
    Bendr_contextualizer_path: str = '/pretrained_weights/Bendr/encoder.pt'
    SppEEGNet_path: str = '/pretrained_weights/SppEEGNet/tuh_all_ckp.pt'
    Mbrain_path: str = '/pretrained_weights/Mbrain/01_02_06'
    BFM_path: str = '/pretrained_weights/BFM'
    CBraMod_path: str = '/pretrained_weights/CBraMod/pretrained_weights.pth'
    NeuroGPT_path: str = '/pretrained_weights/NeuroGPT/pytorch_model.bin'
    NeuroLM_path: str = '/pretrained_weights/NeuroLM/NeuroLM-B.pt'
    NeuroLM_token_path: str = '/pretrained_weights/NeuroLM/VQ.pt'
    GPT2_folder_path: str = '/pretrained_weights/GPT2/gpt2'
    EEGPT_path: str = '/pretrained_weights/EEGPT/eegpt_mcae_58chs_4s_large4E.ckpt'
    BrainOmni_path: str = '/pretrained_weights/BrainOmni/base'
    BrainOmni_tokenizer_path: str = '/pretrained_weights/BrainOmni/braintokenizer'
    REVE_path: str = '/pretrained_weights/REVE/REVE_base'
    REVE_pos_path: str = '/pretrained_weights/REVE/REVE_pos'
