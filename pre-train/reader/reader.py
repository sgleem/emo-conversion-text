import torch
import torch.utils.data
import random
import numpy as np
from .symbols import ph2id
import os
from torch.utils.data import DataLoader

def read_text_o(fn):
    text = []
    with open(fn) as f:
        lines = f.readlines()
        lines = lines[1:]
        for line in lines:
            phone = line.strip().split()
            text.append([float(0), int(0), phone])
    return text

def read_text(fn):
    '''
    read phone alignments from file of the format:
    start end phone
    '''
    text = []
    with open(fn) as f:
        lines = f.readlines()
        lines = [l.strip().split() for l in lines]

    for i in lines:
        text.extend(i)
        #for line in lines:
            #start, end, phone = line.strip().split()
            #text.append([int(start), int(end), phone])
    return text

class TextMelIDLoader(torch.utils.data.Dataset):
    
    def __init__(self, list_file, mean_std_file,shuffle=True, pids=None):
        
        file_path_list = []

        with open(list_file) as f:
            lines = f.readlines()
            for line in lines:
                path, n_frame, _ = line.strip().split()
                #speaker_id = path.split('/')[-3].split('_')[2]
                speaker_id = path.split('/')[-2]
                #path = path.replace('data07','home')
                #path = path.replace('CMU_ARCTIC','non')

                #if not speaker_id in pids:
        
                #    continue

                if int(n_frame) >= 800:
                    continue
                
                file_path_list.append(path)


        random.seed(1234)
        if shuffle:
            random.shuffle(file_path_list)
        
        self.file_path_list = file_path_list

        self.mel_mean_std = np.float32(np.load(mean_std_file))
        self.spc_mean_std = np.float32(np.load(mean_std_file.replace('mel_mean', 'spec_mean')))
        #self.sp2id = {speaker_A:0,speaker_B:1}
        #self.sp2id = {'slt': 0, 'rms': 1}
        # self.sp2id={'Neutral':0,'Happy':1,'Sad':2,'Angry':3,'Surprise':4}
        splist=[
            "p225","p226","p227","p228","p229","p230","p231","p232","p233","p234","p236","p237","p238","p239","p240","p241","p243","p244","p245","p246","p247","p248","p249","p250","p251","p252","p253","p254","p255","p256","p257","p258","p259","p260","p261","p262","p263","p264","p265","p266","p267","p268","p269","p270","p271","p272","p273","p274","p275","p276","p277","p278","p279","p280","p281","p282","p283","p284","p285","p286","p287","p288","p292","p293","p294","p295","p297","p298","p299","p300","p301","p302","p303","p304","p305","p306","p307","p308","p310","p311","p312","p313","p314","p316","p317","p318","p323","p326","p329","p330","p333","p334","p335","p336","p339","p340","p341","p343","p345","p347","p351","p360","p361","p362","p363","p364","p374","p376"
        ]
        self.sp2id={
            spk_id: spk_idx for spk_idx, spk_id in enumerate(splist)

        }

    
    def get_path_id(self, path):
        # Custom this function to obtain paths and speaker id
        # Deduce filenames
        text_path = path.replace("/mel_spec/", "/phones/").replace(".npy", ".txt")#.replace(".mel.npy", ".phones").replace("/wav48/", "/txt/")
        # text_path = path.replace('/CMU_ARCTIC', '').replace('/mel', '/txt').replace('.mel.npy', '.phones')
        # b = text_path.split('/')[-1]
        # text_path = os.path.join('/home/zhoukun/nonparaSeq2seqVC_code-master/0013/txt',b)

        mel_path = path #.replace('spec', 'mel')
        #speaker_id = path.split('/')[-3].split('_')[2]
        speaker_id = path.split('/')[-2]
        # use non-trimed version #
        spec_path = path#.replace('mel.npy', 'spec.npy')
        #text_path = text_path.replace('text_trim', 'text')
        #mel_path = mel_path.replace('mel_trim', 'mel')
        #speaker_id = path.split('/')[-3].split('_')[2]
        speaker_id = path.split('/')[-2]

        return mel_path, spec_path, text_path, speaker_id

    def get_text_mel_id_pair(self, path):
        '''
        text_input [len_text]
        text_targets [len_mel]
        mel [mel_bin, len_mel]
        speaker_id [1]
        '''

        mel_path, spec_path, text_path, speaker_id  = self.get_path_id(path)
        # Load data from disk
        text_input = self.get_text(text_path)
        mel = np.load(mel_path).squeeze(0).T
        spc = np.load(spec_path).squeeze(0).T
        speaker_id = [self.sp2id[speaker_id]]
        # Normalize audio 
        mel = (mel - self.mel_mean_std[0])/ self.mel_mean_std[1]
        spc = (spc - self.spc_mean_std[0]) / self.spc_mean_std[1]
        # Format for pytorch
        text_input = torch.LongTensor(text_input)
        mel = torch.from_numpy(np.transpose(mel))
        spc = torch.from_numpy(np.transpose(spc))
        speaker_id = torch.LongTensor(speaker_id)

        return (text_input, mel, spc, speaker_id)
        
    def get_text(self,text_path):

        text = read_text(text_path)
        text_input = []

        #for start, end, ph in text:
            ##dur = int((end - start) / 125000. + 0.6)
            #text_input.append(ph2id[ph])
        for ph in text:
            if ph in ["?", "!", "-", "_"]:
                continue
            text_input.append(ph2id[ph])

        return text_input

    def __getitem__(self, index):
        return self.get_text_mel_id_pair(self.file_path_list[index])

    def __len__(self):
        return len(self.file_path_list)


class TextMelIDCollate():

    def __init__(self, n_frames_per_step=2):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        '''
        batch is list of (text_input, mel, spc, speaker_id)
        '''

        text_lengths = torch.IntTensor([len(x[0]) for x in batch])
        mel_lengths = torch.IntTensor([x[1].size(1) for x in batch])
        mel_bin = batch[0][1].size(0)
        spc_bin = batch[0][2].size(0)

        max_text_len = torch.max(text_lengths).item()
        max_mel_len = torch.max(mel_lengths).item()
        if max_mel_len % self.n_frames_per_step != 0:
            max_mel_len += self.n_frames_per_step - max_mel_len % self.n_frames_per_step
            assert max_mel_len % self.n_frames_per_step == 0

        text_input_padded = torch.LongTensor(len(batch), max_text_len)
        mel_padded = torch.FloatTensor(len(batch), mel_bin, max_mel_len)
        spc_padded = torch.FloatTensor(len(batch), spc_bin, max_mel_len)

        speaker_id = torch.LongTensor(len(batch))
        stop_token_padded = torch.FloatTensor(len(batch), max_mel_len)

        text_input_padded.zero_()
        mel_padded.zero_()
        spc_padded.zero_()
        speaker_id.zero_()
        stop_token_padded.zero_()

        for i in range(len(batch)):
            text =  batch[i][0]
            mel = batch[i][1]
            spc = batch[i][2]

            text_input_padded[i,:text.size(0)] = text 
            mel_padded[i,  :, :mel.size(1)] = mel
            spc_padded[i,  :, :spc.size(1)] = spc
            speaker_id[i] = batch[i][3][0]
            #make sure the downsampled stop_token_padded have the last eng flag 1. 
            stop_token_padded[i, mel.size(1)-self.n_frames_per_step:] = 1 


        return text_input_padded, mel_padded, spc_padded, speaker_id, \
                    text_lengths, mel_lengths, stop_token_padded
