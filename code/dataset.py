import torch
import os, time
import numpy as np
from torch.utils.data import Dataset, DataLoader
from feature_extractor import extract_features

class AudioSet(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.fn_list = list(filter(lambda k: '.npz' in k, os.listdir(self.root_dir)))

    def __len__(self):
        return len(self.fn_list)

    def __getitem__(self, idx):
        fp = os.path.join(self.root_dir, self.fn_list[idx])
        sample = dict(np.load(fp))
        return sample

class AudioSet2(Dataset):
    def __init__(self, samples_npy):
        self.samples = np.load(samples_npy)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fn, label = self.samples[idx]
        label = int(label)
        audio_feat, video_feat = extract_features(fn, label, save=False)
        sample = {'audio':audio_feat, 'video':video_feat, 'label':label}
        return sample

if __name__ == "__main__":
    st = time.time()
    # data_dir = '../data/train/feature/melspec_demo'
    samples_npy = '../data/train/samples.npy'
    audioset = AudioSet2(samples_npy)
    dataloader = DataLoader(audioset, batch_size=5, shuffle=True, num_workers=4)
    for e in range(50):
        for i_batch, batch in enumerate(dataloader):
            print(batch['audio'].size(), batch['video'].size(), batch['label'])

    print('Used time:', time.time()-st)
