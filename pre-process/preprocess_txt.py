import os
import json
# from g2p_en import G2p
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

# g2p=G2p()
vctk_txt_list = glob.glob(VCTK_PATH+"/*/*.txt")

# def extract_phonemes(filename):
#     from phonemizer.phonemize import phonemize
#     from phonemizer.backend import FestivalBackend
#     from phonemizer.separator import Separator
    
#     with open(filename) as f:
#         text=f.read()
#         phones = phonemize(text,
#                            language='en-us',
#                            backend='festival',
#                            separator=Separator(phone=' ',
#                                                syllable='',
#                                                word='')
#         )
#     filename = filename.replace("/dataset00/orig/VCTK/0.80", "dataset02/proc/sleem/VCTK-for-TTS")
#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#     with open(filename.replace(".txt", ".phones"), "w") as outfile:
#         print(phones, file=outfile)
from phonemizer.phonemize import phonemize
from phonemizer.backend import FestivalBackend
from phonemizer.separator import Separator
def read_vctk_txt(txt_path):
    with open(txt_path, 'r') as f:
        text = f.readline()
        phones = phonemize(text,
                            language='en-us',
                            backend='festival',
                            separator=Separator(phone=' ',
                                                syllable='',
                                                word='')
            )
    return phones
def save_phones(txt_path, inp_path_prefix, corpus_type):
    phones = read_vctk_txt(txt_path)
    # phone_txt = post_process_txt(phone_list)
    end_path = txt_path.replace(inp_path_prefix, "")
    output_path = OUT_PATH+"/"+corpus_type+"/"+end_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        print(phones, file=f)
# for vctk_txt_path in tqdm(vctk_txt_list):
#     save_phones(vctk_txt_path, VCTK_PATH, "VCTK")
def read_esd_txt(speaker):
    txt_path = os.path.join(ESD_PATH, speaker, speaker+".txt")
    phones_dict = dict()
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            utt_id = line.split("\t")[0]
            print(utt_id)
            text = line.split("\t")[1]
            phones = phonemize(text,
                                language='en-us',
                                backend='festival',
                                separator=Separator(phone=' ',
                                                    syllable='',
                                                    word='')
                )
            phones_dict[utt_id] = phones
    return phones_dict

esd_spk_list = ["00"+str(n) for n in range(11, 21)] # range(11, 21)
phones_dict = read_esd_txt("0011")
for spk_id in esd_spk_list:
    print("Processing", spk_id)
    for utt_id, phones in phones_dict.items():
        true_utt_id = spk_id+"_"+utt_id.split("_")[-1]
        txt_path = os.path.join(OUT_PATH, "ESD", spk_id, true_utt_id+".txt")
        
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, 'w') as f:
            print(phones, file=f)