import os

seen_speakers = []
ENV_PATH="../config/msp.json"
import json
with open(ENV_PATH, 'r') as f:
    env = json.load(f)
MEL_PATH = env["MEL_PATH"]+"/VCTK"
for seen_speaker in os.listdir(MEL_PATH):
    if seen_speaker=="p315":
        continue
    seen_speakers.append(seen_speaker)
seen_speakers.sort()
seen_speakers = seen_speakers[:99]


phone_list = ['AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
            'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
            'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
            'EY2', 'F', 'G', 'HH',
            'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
            'M', 'N', 'NG', 'OW0', 'OW1',
            'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
            'UH0', 'UH1', 'UH2', 'UW',
            'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
phone_list += ["<pad>", "<unk>", "<s>", "</s>"]

#seen_speakers = ['p336', 'p240', 'p262', 'p333', 'p297', 'p339', 'p276', 'p269', 'p303', 'p260', 'p250', 'p345', 'p305', 'p283', 'p277', 'p302', 'p280', 'p295', 'p245', 'p227', 'p257', 'p282', 'p259', 'p311', 'p301', 'p265', 'p270', 'p329', 'p362', 'p343', 'p246', 'p247', 'p351', 'p263', 'p363', 'p249', 'p231', 'p292', 'p304', 'p347', 'p314', 'p244', 'p261', 'p298', 'p272', 'p308', 'p299', 'p234', 'p268', 'p271', 'p316', 'p287', 'p318', 'p264', 'p313', 'p236', 'p238', 'p334', 'p312', 'p230', 'p253', 'p323', 'p361', 'p275', 'p252', 'p374', 'p286', 'p274', 'p254', 'p310', 'p306', 'p294', 'p326', 'p225', 'p255', 'p293', 'p278', 'p266', 'p229', 'p335', 'p281', 'p307', 'p256', 'p243', 'p364', 'p239', 'p232', 'p258', 'p267', 'p317', 'p284', 'p300', 'p288', 'p341', 'p340', 'p279', 'p330', 'p360', 'p285']

ph2id = {ph:i for i, ph in enumerate(phone_list)}
id2ph = {i:ph for i, ph in enumerate(phone_list)}
sp2id = {sp:i for i, sp in enumerate(seen_speakers)}
id2sp = {i:sp for i, sp in enumerate(seen_speakers)}
