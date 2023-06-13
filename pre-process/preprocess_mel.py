import os
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
import librosa
import json
import numpy as np

# conda activate bigvgan

# VCTK_PATH= "/mnt/deeplearning/dataset00/orig/VCTK/0.80/wav48"
# ESD_PATH = "/mnt/deeplearning/datasetext00/orig/ESD"
# OUT_PATH = "/mnt/deeplearning/datasetext00/proc/sleem/mel_spec"

ENV_PATH="../config/msp.json"
with open(ENV_PATH, 'r') as f:
    env = json.load(f)
VCTK_PATH = env["VCTK_WAV_PATH"]
ESD_PATH = env["ESD_PATH"]
OUT_PATH = env["MEL_PATH"]


# Load bigvgan setting
# Reference: https://github.com/NVIDIA/BigVGAN
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
with open("mel_config.json", 'r') as f:
    h = json.load(f)
    h = AttrDict(h)

def extract_mel(filename, tmp_out=None):
    import torch
    from meldataset import mel_spectrogram
    wav, sr = librosa.load(filename, sr=h.sampling_rate, mono=True)
    wav = torch.FloatTensor(wav)
    # print(wav)
    # compute mel spectrogram from the ground truth audio
    x = mel_spectrogram(wav.unsqueeze(0), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
    x = x.numpy().astype(np.float32)
    # Make sure to apply exp after conversion!!!!
    # x = np.log(x.numpy()).astype(np.float32)
    return x

# Mel spec extractor
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
def extract_all_mel_spec(wav_list):
    with Pool(cpu_count()) as p:
        mel_list = list(tqdm(p.imap(extract_mel, wav_list), total=len(wav_list)))
    return mel_list

# Extract VCTK
import glob
import sys
def extract_feat(wav_list, inp_path_prefix, corpus_type):
    mel_list = extract_all_mel_spec(wav_list)
    for input_file_path, cur_mel in zip(wav_list, mel_list):
        end_path = input_file_path.replace(inp_path_prefix, "").replace(".wav", ".npy")
        output_path = OUT_PATH+"/"+corpus_type+"/"+end_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, cur_mel)

vctk_wav_list = glob.glob(VCTK_PATH+"/*/*.wav")
extract_feat(vctk_wav_list, VCTK_PATH, "VCTK")
# Extract ESD
esd_spk_id = ["00"+str(n) for n in range(11, 21)]
esd_wav_list = []
for spk_id in esd_spk_id:
    cur_wav_list = glob.glob(ESD_PATH+"/"+spk_id+"/*/*/*.wav")
    esd_wav_list.extend(cur_wav_list)
extract_feat(esd_wav_list, ESD_PATH, "ESD")



def estimate_mean_std(root, corpus_type, num=2000):
    '''
    use the training data for estimating mean and standard deviation
    use $num utterances to avoid out of memory
    '''
    specs, mels = [], []
    counter_sp, counter_mel = 0, 0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.npy') and counter_mel<num:
                path = os.path.join(dirpath, f)
                mels.append(np.load(path).squeeze(0).T)
                counter_mel += 1
    
    # specs = np.vstack(specs)
    mels = np.vstack(mels)
    specs = mels

    mel_mean = np.mean(mels,axis=0)
    mel_std = np.std(mels, axis=0)
    spec_mean = np.mean(specs, axis=0)
    spec_std = np.std(specs, axis=0)

    out_root = OUT_PATH+"/norm/"+corpus_type
    os.makedirs(out_root, exist_ok=True)
    np.save(os.path.join(out_root,"spec_mean_std.npy"),
        [spec_mean, spec_std])
    np.save(os.path.join(out_root,"mel_mean_std.npy"),
        [mel_mean, mel_std])

print(OUT_PATH)
for corpus_type in ["VCTK", "ESD"]:
    estimate_mean_std(OUT_PATH+"/"+corpus_type, corpus_type)