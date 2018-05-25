import numpy as np 
import multiprocessing as mp
import librosa
import os, json, time
import cv2

data_dir = '../data/train'
ontology_fp = '../data/ontology/small_ontology.json'
labels_fp = '../data/ontology/small_label.json'
audio_dir = os.path.join(data_dir, 'audio')
video_dir = os.path.join(data_dir, 'video')
video_width = 240

def get_intersection(audio_list, video_list):
    al = set([a.split('.')[0] for a in audio_list])
    vl = set([v.split('.')[0] for v in video_list])
    return list(al & vl)

def extract_audio(fn):
    fp = os.path.join(audio_dir, fn+'.mp3')
    y, sr = librosa.load(fp) # sr = 22050
    melspec = librosa.feature.melspectrogram(y, sr)
    return melspec # should be in shape (F, T) = (128, T)

def extract_video(fn):
    fp = os.path.join(video_dir, fn+'.mp4')
    cap = cv2.VideoCapture(fp)
    frames = []
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        frame = frame.transpose((2,1,0))
        # Crop width
        if frame.shape[1] > video_width:
            start_w = (frame.shape[1] - video_width) // 2
            frame = frame[:, start_w:start_w+video_width, :]
        # Crop height
        if frame.shape[2] > video_width:
            start_w = (frame.shape[2] - video_width) // 2
            frame = frame[:, start_w:start_w+video_width, :]
        frames.append(frame)
    frames = np.array(frames) # should be in shape (N, C, W, H) = (100, 3, 240, 240)
    return frames

def extract_features(fn, label):
    print('Extracting {}...'.format(fn))
    audio_feat = extract_audio(fn)
    video_feat = extract_video(fn)
    if (audio_feat is not None) and (video_feat is not None):
        feat_fp = os.path.join(feat_dir, fn)
        np.savez(feat_fp, audio=audio_feat, video=video_feat, label=label)

def extract(feat_type='melspec'):
    global feat_dir
    feat_dir = os.path.join(data_dir, 'feature', feat_type)
    if not os.path.isdir(feat_dir):
        os.makedirs(feat_dir)

    audio_list = os.listdir(audio_dir)
    video_list = os.listdir(video_dir)
    data_list = get_intersection(audio_list, video_list)
    print('{} pairs of data. ({} audio, {} video)'.format(len(data_list), len(audio_list), len(video_list)))

    sample_info = np.load(os.path.join(data_dir, 'train_samples.npy')).item()
    samples = []
    too_much_label = 0
    with open(labels_fp) as f:
        labels = json.load(f)
        label_dict = {l: labels.index(l) for l in labels}
    for fn in data_list:
        try:
            ytid = fn[3:14]
            _, st, et = fn[14:].split('_')
            cids = sample_info[ytid]['cids']
            # Currently we discard data with more then one label.
            if len(cids) != 1:
                too_much_label += 1
                continue
            idx = label_dict[cids[0]]
            samples.append((fn, idx))
        except Exception as err:
            print(fn, 'Error:', err)
            continue
    print('{} samples of data. ({} samples have more then 1 label)'.format(len(samples), too_much_label))

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(processes=8)
    results = pool.starmap(extract_features, samples)
    failed_list = list(filter(lambda a: a != None, results))
    q.put('kill')
    pool.close()



if __name__ == "__main__":
    start_time = time.time()
    extract()
    end_time = time.time() - start_time
    print('Total spend: {} mins'.format(end_time/60))
