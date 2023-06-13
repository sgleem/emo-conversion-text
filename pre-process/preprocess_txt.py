import os
import json
from g2p_en import G2p
import numpy as np
import glob
from tqdm import tqdm
import sys
# conda activate bigvgan

# VCTK_PATH= "/mnt/deeplearning/dataset00/orig/VCTK/0.80/txt"
# ESD_PATH = "/mnt/deeplearning/datasetext00/orig/ESD"
# OUT_PATH = "/mnt/deeplearning/datasetext00/proc/sleem/phones"

ENV_PATH="../config/msp.json"
with open(ENV_PATH, 'r') as f:
    env = json.load(f)
VCTK_PATH = env["VCTK_TXT_PATH"]
ESD_PATH = env["ESD_PATH"]
OUT_PATH = env["PHONES_PATH"]

g2p=G2p()
vctk_txt_list = glob.glob(VCTK_PATH+"/*/*.txt")
def read_vctk_txt(txt_path):
    with open(txt_path, 'r') as f:
        txt = f.readline()
    out = g2p(txt)
    return out
def post_process_txt(phone_list):
    out_list = []
    for phone in phone_list:
        if phone not in  [" ", ",", ".", "'"]:
            out_list.append(phone)
    return " ".join(out_list)
def save_phones(txt_path, inp_path_prefix, corpus_type):
    phone_list = read_vctk_txt(txt_path)
    phone_txt = post_process_txt(phone_list)
    end_path = txt_path.replace(inp_path_prefix, "")
    output_path = OUT_PATH+"/"+corpus_type+"/"+end_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(phone_txt)
for vctk_txt_path in tqdm(vctk_txt_list):
    save_phones(vctk_txt_path, VCTK_PATH, "VCTK")

