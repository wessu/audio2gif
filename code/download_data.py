# Path to ffmpeg
ffmpeg_path = '../bin/ffmpeg/ffmpeg'
ontology_fp = '../data/ontology/small_ontology.json'
data_list_fp = '../data/unbalanced_train_segments.csv'
train_data_dir = '../data/train/'

import sys
import os.path
import json
from collections import namedtuple, Counter
import multiprocessing as mp
# Make sure ffmpeg is on the path so sk-video can find it
sys.path.append(os.path.dirname(ffmpeg_path))
import skvideo.io
# import cv2
# import matplotlib.pyplot as plt
import numpy as np
import pafy
# import soundfile as sf
import subprocess as sp

Sample = namedtuple('Sample', ['ytid', 'start_t', 'end_t', 'cids'])

# Set output settings
audio_codec = 'mp3'
audio_container = 'mp3'
video_codec = 'h264'
video_container = 'mp4'

def select_data(id_list):
    id_set = set(id_list)
    # Load the AudioSet training set
    with open(data_list_fp) as f:
        lines = f.readlines()

    dt_list = [line.strip().split(',', 3) for line in lines[3:]]
    print("Totally {} samples in {}".format(len(dt_list), data_list_fp))
    samples = []
    cnt = Counter()
    for dt in dt_list:
        cids = list(set(dt[3][2:-1].split(',')) & id_set)
        if len(cids) > 0:
            samples.append(Sample(dt[0], float(dt[1]), float(dt[2]), cids))
            cnt[len(cids)] += 1

    print("Number of labels:")
    print(cnt)

    return samples

def download_data(save_dir, samples):

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    original_stdout = sys.stdout
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(processes=8)
    args = [(save_dir, idx, len(samples), sample) for idx, sample in enumerate(samples)]
    results = pool.starmap(download_one_sample, args)
    # results = [pool.apply(download_one_sample, args=(save_dir, idx, len(samples), sample)) for idx, sample in enumerate(samples)]
    failed_list = list(filter(lambda a: a != None, results))

    q.put('kill')
    pool.close()
    sys.stdout = original_stdout

    print("{} samples are successfully downloaded. {} samples failed.".format(len(samples)-len(failed_list),len(failed_list)))
    if len(failed_list) > 0:
        np.savetxt(os.path.join(save_dir, 'failed_download_list.txt'), failed_list, fmt='%s')

def download_one_sample(save_dir, idx, n_samples, sample):
    # print("Running on {}".format(os.getpid()))
    print("{}/{}. YouTube ID: {}. Window: ({}, {})".format(idx+1, n_samples, sample.ytid, sample.start_t, sample.end_t))

    result = None
    try:
        # Get the URL to the video page
        video_page_url = 'https://www.youtube.com/watch?v={}'.format(sample.ytid)

        # Get the direct URLs to the videos with best audio and with best video (with audio)
        video = pafy.new(video_page_url)

        best_video = video.getbestvideo()
        best_video_url = best_video.url
        # print("Video URL: " + best_video_url)

        best_audio = video.getbestaudio()
        best_audio_url = best_audio.url
        # print("Audio URL: " + best_audio_url)

        # Get output video and audio filepaths
        duration = sample.end_t - sample.start_t
        basename_fmt = 'yt_{}_{}_{}'.format(sample.ytid, int(sample.start_t), int(sample.end_t))
        video_filepath = os.path.join(save_dir, basename_fmt + '.' + video_container)
        audio_filepath = os.path.join(save_dir, basename_fmt + '.' + audio_codec)


        # Download the video
        video_dl_args = [ffmpeg_path, '-n',
            '-ss', str(int(sample.start_t)),   # The beginning of the trim window
            '-i', best_video_url,   # Specify the input video URL
            '-t', str(duration),    # Specify the duration of the output
            '-f', video_container,  # Specify the format (container) of the video
            '-r', '10',     # Specify the framerate
            '-vcodec', 'h264',      # Specify the output encoding
            '-vf', 'scale=w=320:h=240',
            video_filepath]

        proc = sp.Popen(video_dl_args, stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print("{}. {}".format(idx+1, stderr))
        else:
            print("{}. Downloaded video to {}".format(idx+1, video_filepath))

        # Download the audio
        audio_dl_args = [ffmpeg_path, '-n',
            '-ss', str(int(sample.start_t)),    # The beginning of the trim window
            '-i', best_audio_url,    # Specify the input video URL
            '-t', str(duration),     # Specify the duration of the output
            '-vn',                   # Suppress the video stream
            '-ac', '2',              # Set the number of channels
            # '-sample_fmt', 's16',    # Specify the bit depth
            '-acodec', audio_codec,  # Specify the output encoding
            '-ar', '44100',          # Specify the audio sample rate
            audio_filepath]

        proc = sp.Popen(audio_dl_args, stdout=sp.PIPE, stderr=sp.PIPE)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print("{}. {}".format(idx+1, stderr))
            result = sample.ytid
        else:
            print("{}. Downloaded audio to {}".format(idx+1, audio_filepath))
    except Exception as excp:
        print(type(excp))
        print("{}. {}".format(idx+1, excp))
        result = sample.ytid
    print("")
    return result

def save_samples(samples):
    sample_dict = {s.ytid: {'start_t':s.start_t, 'end_t':s.end_t, 'cids':s.cids} for s in samples}
    np.save('../data/train_samples', sample_dict)


if __name__ == "__main__":
    with open(ontology_fp) as f:
        ont_dict = json.load(f)
    print("Selected labels:")
    print(ont_dict)
    id_list = ont_dict.values()
    samples = select_data(id_list)
    save_samples(samples)
    # download_data(train_data_dir, samples)