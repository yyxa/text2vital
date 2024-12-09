from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np
import psutil

model = AutoModel.from_pretrained(
    './OpenJMLA', 
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    offload_folder="./offload"
)

device = model.device
# sample rate: 16k
music_path = './4.mp3'
# 1. get logmelspectrogram
# get the file wav_to_mel.py from https://github.com/taugastcn/SpectPrompt.git
from wav_to_mel import wav_to_mel
lms = wav_to_mel(music_path)

import os
from torch.nn.utils.rnn import pad_sequence
import random
# get the file transforms.py from https://github.com/taugastcn/SpectPrompt.git
from transforms import Normalize, SpecRandomCrop, SpecPadding, SpecRepeat
transforms = [ Normalize(-4.5, 4.5), SpecRandomCrop(target_len=2992), SpecPadding(target_len=2992), SpecRepeat() ]
lms = lms.numpy()
for trans in transforms:
    lms = trans(lms)

# 2. template of input
input_dic = dict()
input_dic['filenames'] = [music_path.split('/')[-1]]
input_dic['ans_crds'] = [0]
input_dic['audio_crds'] = [0]
input_dic['attention_mask'] = torch.tensor([[1, 1, 1, 1, 1]]).to(device)
input_dic['input_ids'] = torch.tensor([[1, 694, 5777, 683, 13]]).to(device)
input_dic['spectrogram'] = torch.from_numpy(lms).unsqueeze(dim=0).to(device)
# 3. generation
model.eval()
# gen_ids = model.forward_test(input)
gen_ids = model.forward_test(input_dic)
gen_text = model.neck.tokenizer.batch_decode(gen_ids.clip(0))
# 4. Post-processing
# Given that the training data may contain biases, the generated texts might need some straightforward post-processing to ensure accuracy.
# In future versions, we will enhance the quality of the data.
print(gen_text)
# gen_text = gen_text.split('<s>')[-1].split('\n')[0].strip()
# gen_text = gen_text.replace(' in Chinese','')
# gen_text = gen_text.replace(' Chinese','')
# print(gen_text)
