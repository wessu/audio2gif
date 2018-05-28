import numpy as np 
import multiprocessing as mp
import librosa
import os, json, time
import cv2

data_dir = '../data/train'
audio_dir = os.path.join(data_dir, 'audio')
video_dir = os.path.join(data_dir, 'video')