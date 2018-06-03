import numpy as np
import os
import multiprocessing as mp

data_dir = '../data/audioset/train/feature/melspec'
save_dir = os.path.join(data_dir, 'wrap')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# fn_list = list(filter(lambda k: '.npz' in k, os.listdir(data_dir)))
# np.random.shuffle(fn_list)
n = 9600

# def demo():
#     for fn in os.listdir(data_dir):
#         fp = os.path.join(data_dir, fn)
#         features = np.load(fp)
#         audio_feature = features['audio'] # Current shape: (128, 431). Type: float
#         video_feature = features['video'] # Current shape: (100, 3, 240, 240). Type: uint8
#         label = features['label'].item()  # A scalar. Type: int

# def combine(i):
#     feat_list = []
#     fl = fn_list[i*n:(i+1)*n]
#     fl_fp = os.path.join(save_dir, 'wrap_{}_fl'.format(i))
#     np.save(fl_fp, fl)
#     for fn in fl:
#         fp = os.path.join(data_dir, fn)
#         features = dict(np.load(fp))
#         features.pop('video')
#         feat_list.append(features)
#         if len(feat_list) % 1000 == 0:
#             print('append {} to {}. ({}/{})'.format(fn, i, len(feat_list), len(fl)))
#     save_fp = os.path.join(save_dir, 'wrap_{}'.format(i))
#     np.save(save_fp, feat_list)
#     print('wrap_{} saved'.format(i))

def combine(i):
    feat_list = []
    fl_fp = os.path.join(save_dir, 'wrap_{}_fl.npy'.format(i))
    fl = np.load(fl_fp)
    print(len(fl))
    for fn in fl:
        fp = os.path.join(data_dir, fn)
        features = dict(np.load(fp))
        features.pop('video')
        feat_list.append(features)
        if len(feat_list) % 500 == 0:
            print('append {} to {}. ({}/{})'.format(fn, i, len(feat_list), len(fl)))
    save_fp = os.path.join(save_dir, 'wrap_{}.npy'.format(i))
    np.save(save_fp, feat_list)
    print('wrap_{} saved'.format(i))

# print('Wrap', data_dir, 'into', save_dir)
# m = int(np.ceil(len(fn_list)/float(n)))
# print('{} / {} = {}'.format(len(fn_list), n, m))
# manager = mp.Manager()
# q = manager.Queue()
# pool = mp.Pool(processes=2)
# results = pool.map(combine, [2,4])
# q.put('kill')
# pool.close()

# print('Finish!')

for fn in ['wrap_0.npy', 'wrap_1.npy', 'wrap_2.npy', 'wrap_4.npy', 'wrap_3.npy', 'wrap_5.npy']:
    print(fn)
    if 'fl' in fn:
        continue
    try:
        w = np.load(os.path.join(save_dir, fn))
        print(len(w))
        print(w.shape)
        print(w[0])
        print(w[-1])
        for _ in range(3):
            print(w[np.random.randint(0,len(w))])
    except Exception as err:
        print(fn, 'Error:', err)
    print()

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
