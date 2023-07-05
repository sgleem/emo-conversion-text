import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt


import os
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader

from reader import TextMelIDLoader, TextMelIDCollate, id2ph, id2sp
from hparams import create_hparams
from model import Parrot, lcm
from train import load_model
import scipy.io.wavfile


########### Configuration ###########
hparams = create_hparams()

# use seen (tlist) or unseen list (hlist)
test_list = "../data/esd_list_0011/testing_mel_list.txt"
checkpoint_path='emoVC_for_0011/checkpoint_2299'# TTS or VC task?
emb_path = 'emoVC_for_0011/embeddings'
target_emo = 'Angry'
input_text=False
# number of utterances for generation
NUM=10
ISMEL=(not hparams.predict_spectrogram)
#####################################

model = load_model(hparams)

model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.eval()

test_set = TextMelIDLoader(test_list, hparams.mel_mean_std, shuffle=True)
sample_list = test_set.file_path_list
collate_fn = TextMelIDCollate(lcm(hparams.n_frames_per_step_encoder,
                        hparams.n_frames_per_step_decoder))

test_loader = DataLoader(test_set, num_workers=1, shuffle=False,
                              sampler=None,
                              batch_size=1, pin_memory=False,
                              drop_last=True, collate_fn=collate_fn)



task = 'tts' if input_text else 'vc'
path_save = os.path.join(checkpoint_path.replace('checkpoint', 'test-to-'+target_emo), task)
if not os.path.exists(path_save):
    os.makedirs(path_save)

target_emb = np.load(os.path.join(emb_path, target_emo+'.npy'))
target_emb = torch.Tensor(target_emb).cuda().mean(dim=0, keepdim=True)
print(target_emb.size())

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

with torch.no_grad():

    errs = 0
    totalphs = 0

    for i, batch in enumerate(test_loader):
        if i == NUM:
            break
        
        sample_emo = sample_list[i].split('/')[-3]
        sample_id = sample_emo+"_"+sample_list[i].split('/')[-1][:-4]
        print(('%d index %s, decoding ...'%(i,sample_id)))

        x, y = model.parse_batch(batch)
        predicted_mel, post_output, predicted_stop, alignments, \
            text_hidden, audio_seq2seq_hidden, audio_seq2seq_phids, audio_seq2seq_alignments, \
            speaker_id = model.inference(x, input_text, None, hparams.beam_width, speaker_embedding=target_emb)

        post_output = post_output.data.cpu().numpy()[0]
        alignments = alignments.data.cpu().numpy()[0].T
        audio_seq2seq_alignments = audio_seq2seq_alignments.data.cpu().numpy()[0].T

        text_hidden = text_hidden.data.cpu().numpy()[0].T #-> [hidden_dim, max_text_len]
        audio_seq2seq_hidden = audio_seq2seq_hidden.data.cpu().numpy()[0].T
        audio_seq2seq_phids = audio_seq2seq_phids.data.cpu().numpy()[0] # [T + 1]

        task = 'TTS' if input_text else 'VC'

        # recover_wav(post_output, 
        #             os.path.join(path_save, 'Wav_%s_ref_%s_%s.wav'%(sample_id, ref_sp, task)), 
        #             ismel=ISMEL)
        mean, std = np.load(hparams.mel_mean_std)
        post_output = 1.2 * post_output.T * std + mean
        post_output = post_output.T
        post_output_path = os.path.join(path_save, 'Mel_%s_ref_%s_%s.npy'%(sample_id, target_emo, task))
        np.save(post_output_path, post_output)
                
        audio_seq2seq_phids = [id2ph[id] for id in audio_seq2seq_phids[:-1]]
        target_text = y[0].data.cpu().numpy()[0]
        target_text = [id2ph[id] for id in target_text[:]]

        print(audio_seq2seq_phids)
        print(target_text)
       
        err = levenshteinDistance(audio_seq2seq_phids, target_text)
        print(err, len(target_text))

        errs += err
        totalphs += len(target_text)

print(float(errs)/float(totalphs))

        
        
