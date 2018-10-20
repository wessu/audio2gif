from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
import time
import multiprocessing as mp
import torch
from torch.utils.data import Dataset, DataLoader
from feature_extractor import extract_features


from miscc.config import cfg


class GIFDataset(data.Dataset):
    def __init__(self, data_dir, embedding_size, stage=1,  imsize=64, n_frames=4):
        self.imsize = imsize
        self.data_dir = data_dir
        self.filenames = [name for name in os.listdir(data_dir) if not name.startswith('.')]
        print("creating GIF dataset")
        print(self.filenames)
        self.n_gif = len(self.filenames)
        print("with {} data".format(self.n_gif))
        self.n_frames = n_frames
        self.noise_dim = embedding_size
        self.stage = stage

    def get_gif(self, img_name, n_frames):
        gif_base = os.path.join(self.data_dir, img_name)
        frames = os.listdir(gif_base)
        if len(frames) > n_frames:
            frames_order = np.arange(len(frames))
            np.random.shuffle(frames_order)
            selected_frames = frames_order[:n_frames]
            selected_frames = sorted(selected_frames)
        else:
            selected_frames = np.tile(np.arange(len(frames)), self.n_frames//len(frames) + 1)[:n_frames]
        imgs = []
        for f in selected_frames:
            img = Image.open(os.path.join(gif_base, "{}.jpg".format(f)))
            if img.size[0] != self.imsize:
                img = img.resize((self.imsize, self.imsize))
            img = np.array(img.convert('RGB')) # size, size , 3
            imgs.append(img)
        print(np.array(imgs).shape)
        # imgs = D, H, W. C -> C, D, H, W
        return np.array(imgs).transpose((3,0,1,2))

    def get_image(self, img_name):
        return np.squeeze(self.get_gif(img_name, 1)) # return C, H, W

    def get_fake_embedding(self):
        return np.random.randn(self.noise_dim)

    def __len__(self):
        return self.n_gif

    def __getitem__(self, index):
        if self.stage == 2:
            return self.get_gif(self.filenames[index], self.n_frames), self.get_fake_embedding()
        else:
            return self.get_image(self.filenames[index]), self.get_fake_embedding()

class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', embedding_type='cnn-rnn',
                 imsize=64, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.imsize = imsize
        self.data = []
        self.data_dir = data_dir
        if data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        #self.filenames = self.load_filenames(split_dir)
        filenames = os.listdir(os.path.join(data_dir, "images"))[:16]
        self.filenames = [f.strip(".jpg") for f in filenames]
        print(self.filenames[0])
        self.embeddings = self.load_embedding(split_dir, embedding_type)
        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        # self.captions = self.load_all_captions()

    def get_img(self, img_path, bbox):
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        if bbox is not None:
            R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
            center_x = int((2 * bbox[0] + bbox[2]) / 2)
            center_y = int((2 * bbox[1] + bbox[3]) / 2)
            y1 = np.maximum(0, center_y - R)
            y2 = np.minimum(height, center_y + R)
            x1 = np.maximum(0, center_x - R)
            x2 = np.minimum(width, center_x + R)
            img = img.crop([x1, y1, x2, y2])
        load_size = int(self.imsize * 76 / 64)
        img = img.resize((load_size, load_size), PIL.Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in xrange(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_all_captions(self):
        caption_dict = {}
        for key in self.filenames:
            caption_name = '%s/text/%s.txt' % (self.data_dir, key)
            captions = self.load_captions(caption_name)
            caption_dict[key] = captions
        return caption_dict

    def load_captions(self, caption_name):
        cap_path = caption_name
        with open(cap_path, "r") as f:
            captions = f.read().decode('utf8').split('\n')
        captions = [cap.replace("\ufffd\ufffd", " ")
                    for cap in captions if len(cap) > 0]
        return captions

    def load_embedding(self, data_dir, embedding_type):
        if embedding_type == 'cnn-rnn':
            embedding_filename = '/char-CNN-RNN-embeddings.pickle'
        elif embedding_type == 'cnn-gru':
            embedding_filename = '/char-CNN-GRU-embeddings.pickle'
        elif embedding_type == 'skip-thought':
            embedding_filename = '/skip-thought-embeddings.pickle'

        with open(data_dir + embedding_filename, 'rb') as f:
            embeddings = pickle.load(f, fix_imports=True, encoding='latin1')
            embeddings = np.array(embeddings)
            # embedding_shape = [embeddings.shape[-1]]
            print('embeddings: ', embeddings.shape)
        return embeddings

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f)
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir):
        filepath = os.path.join(data_dir, 'filenames.pickle')
        with open(filepath, 'rb') as f:
            filenames = pickle.load(f)
        print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        return filenames

    def __getitem__(self, index):
        key = self.filenames[index]
        # cls_id = self.class_id[index]
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s/CUB_200_2011' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir

        # captions = self.captions[key]
        embeddings = self.embeddings[index, :, :]
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        img = self.get_img(img_name, bbox)

        embedding_ix = random.randint(0, embeddings.shape[0]-1)
        embedding = embeddings[embedding_ix, :]
        if self.target_transform is not None:
            embedding = self.target_transform(embedding)
        return img, embedding

    def __len__(self):
        return len(self.filenames)


class AudioSet(Dataset):
    def __init__(self, root_dir, frame_hop_size=2, n_frames=5, stage=1):
        self.root_dir = root_dir
        self.fn_list = list(filter(lambda k: '.npy' in k, os.listdir(self.root_dir)))
        # self.fn_list = ['wrap_4.npy', 'wrap_7.npy', 'wrap_8.npy', 'wrap_51.npy', 
                        # 'wrap_15.npy', 'wrap_16.npy', 'wrap_19.npy', 'wrap_32.npy']
        self.frame_hop_size = frame_hop_size
        self.n_frames = n_frames
        self.stage = stage
        if self.stage == 1:
            self.ft_type = 'image'
        elif self.stage == 2:
            self.ft_type = 'video'
        else:
            raise Exception('Stage should be either 1 or 2. Not {}.'.format(self.stage))

        # self.samples = []
        # n_samples = np.zeros(10)
        # for fn in self.fn_list[:8]:
        #     try:
        #         print('Loading', fn)
        #         fp = os.path.join(self.root_dir, fn)
        #         samples = [{self.ft_type: np.swapaxes(s[self.ft_type], 1, 2) if self.ft_type == 'video' else s[self.ft_type], 
        #                     'audio': s['audio'], 
        #                     'label': s['label']} for s in np.load(fp).tolist()]
        #         for s in samples:
        #             n_samples[s['label']] += 1
        #         self.samples += samples
        #     except:
        #         print('Load {} failed'.format(fn))
        print('Number of samples in the dataset: {}'.format(len(self.fn_list)))
        np.savetxt('../data/n_training_samples', len(self.fn_list))

    def __len__(self):
        return len(self.fn_list)

    def __getitem__(self, idx):
        try:
            fp = os.path.join(self.root_dir, self.fn_list[idx])
            sample = np.load(fp)
        except:
            print("Fail to load data from {}".format(self.fn_list[idx]))
            exit(1)
        return (self.sample['audio'], self.sample[self.ft_type][k], self.sample['label'])

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    inputs = torch.cat(inputs)
    targets = torch.cat(targets)
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs), dtype=int)
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs), batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

class AudioSetImage(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.fn_list = list(filter(lambda k: '.npy' in k, os.listdir(self.root_dir)))
        self.ft_type = 'image'
        self.samples = []
        self.current_i = 0
        # for fn in self.fn_list:
        #     try:
        #         fp = os.path.join(self.root_dir, fn)
        #         samples = np.load(fp).tolist()
        #         for s in samples:
        #             s.pop('video')
        #         self.samples += 
        #         print('{} Loaded'.format(fn))
        #     except:
        #         print('Load {} failed'.format(fn))
        for fn in self.fn_list:
            try:
                print('Loading', fn)
                fp = os.path.join(self.root_dir, fn)
                samples = np.load(fp).tolist()
                self.samples += [(s['image'][i], s['label']) for s in samples for i in range(s['image'].shape[0])]
            except:
                print('Load {} failed'.format(fn))


    def __len__(self):
        return len(self.samples)
        # return len(self.fn_list)

    def __getitem__(self, idx):
        # self.samples = []
        # fn = self.fn_list[idx]
        # print('Loading {}...'.format(fn))
        # st = time.time()
        # fp = os.path.join(self.root_dir, fn)
        # samples = np.load(fp).tolist()
        # print('Finished loading {}! ({:.4f} secs)'.format(fn, time.time()-st))
        # # This return a tuple (img, label)
        # self.samples += [(s['image'][i], s['label']) for s in samples for i in range(s['image'].shape[0])]
        return self.samples[idx]

class ImageSet(Dataset):
    def __init__(self, root_dir, fake):
        self.root_dir = root_dir
        self.fn_list = list(filter(lambda k: '.npy' in k, os.listdir(self.root_dir)))
        self.samples = []
        for fn in self.fn_list:
            try:
                print('Loading', fn)
                fp = os.path.join(self.root_dir, fn)
                samples = np.load(fp).tolist()
                if fake:
                    self.samples += [((s['fake']-np.min(s['fake']))*128/np.max(s['fake']), s['label']) for s in samples]
                else:
                    self.samples += [(s['real'], s['label']) for s in samples]
            except:
                print('Load {} failed'.format(fn))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    # def load_wrap(self, fn):
    #     fp = os.path.join(self.root_dir, fn)
    #     samples = np.load(fp)
    #     for sample in samples:
    #         if self.stage == 1:
    #             sample['video'] = self.select_image(sample['video'])
    #         elif self.stage == 2:
    #             sample['video'] = self.select_frames(sample['video'])
    #         elif self.stage == 0: # Train embedding net. Return audio features only.
    #             # sample.pop('video', None)
    #             pass
    #         else:
    #             raise Exception('Stage should be either 1 or 2. Not {}.'.format(self.stage))
    #     return samples

    # def select_frames(self, video):
    #     last_start = video.shape[0] - self.frame_hop_size * self.n_frames
    #     idx = np.random.randint(0, last_start)
    #     gif = video[idx:idx+self.frame_hop_size*self.n_frames:self.frame_hop_size]
    #     return gif

    # def select_image(self, video):
    #     last_start = video.shape[0] - self.frame_hop_size * self.n_frames
    #     idx = np.random.randint(0, last_start)
    #     # idx = 10
    #     img = video[idx].transpose((1, 2, 0))
    #     img = Image.fromarray(img, 'RGB')
    #     img = img.resize((64, 64), Image.BILINEAR)
    #     img = np.array(img.convert('RGB')).transpose((2, 1, 0)) # size, size , 3
    #     return img

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

class AudioSetAudio(Dataset):
    def __init__(self, root_dir, select_data, wrap=True):
        self.samples = []
        if wrap:
            self.root_dir = os.path.join(root_dir, 'wrap')
            self.fn_list = list(filter(lambda k: '.npy' in k, os.listdir(self.root_dir)))
        else:
            self.root_dir = root_dir
            self.fn_list = list(filter(lambda k: '.npz' in k, os.listdir(self.root_dir)))

        print('Loading data...')
        st = time.time()
        
        if wrap:
            for fn in self.fn_list:
                fp = os.path.join(self.root_dir, fn)
                samples = np.load(fp).tolist()
                if select_data:
                    self.samples += [s for s in samples \
                                 if (((s['label'] in [2, 3, 6]) and np.random.rand() < 0.33) or \
                                        (s['label'] not in [2, 3, 6]))]
                else:
                    self.samples += samples
            print('Totally {} files selected'.format(len(self.samples)))
            mm = np.zeros(10)
            for s in self.samples:
                mm[s['label']] += 1
            print('Number of samples in each class')
            print(mm)
            # pool.map(self.load_wrap, self.fn_list)
            # self.samples = np.concatenate(self.samples).tolist()
            # print(len(self.samples))
        else:
            manager = mp.Manager()
            q = manager.Queue()
            pool = mp.Pool(processes=3)
            self.samples = pool.map(self.load, self.fn_list)
            q.put('kill')
            pool.close()
        print('Took %.4f mins to load data' % ((time.time()-st)/60.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def load(self, fn):
        fp = os.path.join(self.root_dir, fn)
        sample = dict(np.load(fp))
        sample.pop('video')
        return sample

    def load_wrap(self, fn):
        fp = os.path.join(self.root_dir, fn)
        samples = np.load(fp)
        self.samples += samples
        return None

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