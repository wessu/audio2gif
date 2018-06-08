import numpy as np
import os, time
import multiprocessing as mp
from PIL import Image

data_dir = '../data/audioset/train/feature/melspec/'
# data_dir = '../data/demo/feature/melspec/'
save_dir = os.path.join(data_dir, 'wrap_all')
fl_dir = os.path.join(data_dir, 'wrap_all_fn_list')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

if not os.path.isdir(fl_dir):
    os.makedirs(fl_dir)

# fn_list = list(filter(lambda k: '.npz' in k, os.listdir(data_dir)))
# np.random.shuffle(fn_list)
n = 1000
k = 3

# def demo():
#     for fn in os.listdir(data_dir):
#         fp = os.path.join(data_dir, fn)
#         features = np.load(fp)
#         audio_feature = features['audio'] # Current shape: (128, 431). Type: float
#         video_feature = features['video'] # Current shape: (100, 3, 240, 240). Type: uint8
#         label = features['label'].item()  # A scalar. Type: int

def combine(i):
    feat_list = []
    save_fp = os.path.join(save_dir, 'wrap_{}'.format(i))
    if os.path.isfile(save_fp+'.npy'):
        print('ignore', i)
        return
    fl = fn_list[i*n:(i+1)*n]
    fl_fp = os.path.join(fl_dir, 'wrap_{}_fl'.format(i))
    np.save(fl_fp, fl)
    # fl = np.load(fl_fp+'.npy')
    for fn in fl:
        fp = os.path.join(data_dir, fn)
        features = dict(np.load(fp))
        features['video'] = np.array([select_frames(features['video']) for _ in range(k)])
        features['image'] = np.array([select_image(features['video'][j]) for j in range(k)])
        feat_list.append(features)
        # if len(feat_list) % 100 == 0:
        #     print('append {} to {}. ({}/{})'.format(fn, i, len(feat_list), len(fl)))
    np.save(save_fp, feat_list)
    print('wrap_{} saved'.format(i))

frame_hop_size = 2
n_frames = 5

def select_frames(video):
    last_start = video.shape[0] - frame_hop_size * n_frames
    idx = np.random.randint(0, last_start)
    gif = video[idx:idx+frame_hop_size*n_frames:frame_hop_size]
    return gif

def select_image(video):
    # last_start = video.shape[0] - frame_hop_size * n_frames
    # idx = np.random.randint(0, last_start)
    # idx = 10
    img = video[0].transpose((1, 2, 0))
    img = Image.fromarray(img, 'RGB')
    img = img.resize((64, 64), Image.BILINEAR)
    img = np.array(img.convert('RGB')).transpose((2, 1, 0)) # size, size , 3
    return img



# def combine(i):
#     feat_list = []
#     fl_fp = os.path.join(save_dir, 'wrap_all_{}.npy'.format(i))
#     fl = np.load(fl_fp)
#     print(len(fl))
#     for fn in fl:
#         fp = os.path.join(data_dir, fn)
#         features = dict(np.load(fp))
#         features.pop('video')
#         feat_list.append(features)
#         if len(feat_list) % 500 == 0:
#             print('append {} to {}. ({}/{})'.format(fn, i, len(feat_list), len(fl)))
#     save_fp = os.path.join(save_dir, 'wrap_{}.npy'.format(i))
#     np.save(save_fp, feat_list)
#     print('wrap_{} saved'.format(i))

# print('Wrap', data_dir, 'into', save_dir)
# fn_list = list(filter(lambda k: '.npz' in k, os.listdir(data_dir)))
# m = int(np.ceil(len(fn_list)/float(n)))
# print('{} / {} = {}'.format(len(fn_list), n, m))


# manager = mp.Manager()
# q = manager.Queue()
# pool = mp.Pool(processes=1)
# results = pool.map(combine, range(m))
# q.put('kill')
# pool.close()

# print('Finish!')

n_samples = np.zeros(10)
for fn in os.listdir(save_dir):
    print(fn)
    if 'fl' in fn:
        continue
    try:
        st = time.time()
        w = np.load(os.path.join(save_dir, fn))
        print('load %.4f secs' % (time.time()-st))
        print(len(w))
        for s in w:
            n_samples[s['label']] += 1
        # print(w.shape)
        # print(w[0].keys())
        # print(w[1]['image'].shape)
        # print(w[-1]['video'].shape)
        # print(w[-2]['audio'].shape)
        # for _ in range(3):
        #     print(w[np.random.randint(0,len(w))]['label'])

    except Exception as err:
        print(fn, 'Error:', err)
    print()
print('Number of samples in training set: ')
print(n_samples)
print('')
print('')

data_dir = '../data/audioset/eval/feature/melspec/'
save_dir = os.path.join(data_dir, 'wrap_all')
n_samples = np.zeros(10)
for fn in os.listdir(save_dir):
    print(fn)
    if 'fl' in fn:
        continue
    try:
        st = time.time()
        w = np.load(os.path.join(save_dir, fn))
        print('load %.4f secs' % (time.time()-st))
        print(len(w))
        for s in w:
            n_samples[s['label']] += 1
        # print(w.shape)
        # print(w[0].keys())
        # print(w[1]['image'].shape)
        # print(w[-1]['video'].shape)
        # print(w[-2]['audio'].shape)
        # for _ in range(3):
        #     print(w[np.random.randint(0,len(w))]['label'])

    except Exception as err:
        print(fn, 'Error:', err)
    print()
print('Number of samples in evaluation set: ')
print(n_samples)

# for i in [2, 4]:
#     feat_list = []
#     try:
#         fl_fp = os.path.join(save_dir, 'wrap_{}_fl.npy'.format(i))
#         fl = np.load(fl_fp)
#         print(len(fl))
#         for fn in fl:
#             fp = os.path.join(data_dir, fn)
#             features = dict(np.load(fp))
#             features.pop('video')
#             feat_list.append(features)
#             if len(feat_list) % 500 == 0:
#                 print('append {} to {}. ({}/{})'.format(fn, i, len(feat_list), len(fl)))
#         save_fp = os.path.join(save_dir, 'wrap_{}.npy'.format(i))
#         np.save(save_fp, feat_list)
#         print('wrap_{} saved'.format(i))
#     except Exception as err:
#         print(fn, 'Error:', err)
#     print()
