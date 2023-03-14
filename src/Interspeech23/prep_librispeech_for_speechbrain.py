import glob
import os
import json
import pandas as pd
import torchaudio
from tqdm import tqdm

ROOT = '/home/mshah1/workhorse3/librispeech/'
SPLITS = ['val', 'train', 'test_clean', 'test_other']
SAMPLERATE = 16_000
for split in SPLITS:
    splitdir = os.path.join(ROOT,split)
    txtfiles = glob.glob(f'{splitdir}/txt/*.txt')
    utt_ids = [os.path.basename(f).split('.')[0] for f in txtfiles]
    wavfiles = [f'{splitdir}/wav/{uid}.wav' for uid in utt_ids]
    
    data = {}
    datalist = []
    t = tqdm(zip(utt_ids, txtfiles, wavfiles))
    t.set_description(split)
    for uid,tf,wf in t:        
        if os.path.exists(tf) and os.path.exists(wf):
            with open(tf) as f:
                txt = f.read()
            duration = torchaudio.info(wf).num_frames / SAMPLERATE
            spk_id = "-".join(uid.split("-")[0:2])
            r = {'ID': uid, 'duration':duration, 'wav':wf, 'spk_id':spk_id, 'wrd':txt}
            data[uid] = r
            datalist.append(r)
    df = pd.DataFrame(datalist)
    df.to_csv(f'{ROOT}/{split}.csv')
    with open(f'{ROOT}/{split}.json', 'w') as f:
        json.dump(data, f)
