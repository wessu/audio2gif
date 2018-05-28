import numpy as np
import os

data_dir = '../data/demo/feature/melspec'

for fn in os.listdir(data_dir):
    fp = os.path.join(data_dir, fn)
    features = np.load(fp)
    audio_feature = features['audio'] # Current shape: (128, 431). Type: float
    video_feature = features['video'] # Current shape: (100, 3, 240, 240). Type: uint8
    label = features['label'].item()  # A scalar. Type: int
