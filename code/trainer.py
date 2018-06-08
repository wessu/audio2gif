from __future__ import print_function
from six.moves import range
from PIL import Image

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time

import numpy as np
import torchfile
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool
from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from miscc.utils import save_img_results, save_model
from miscc.utils import KL_loss
from miscc.utils import compute_discriminator_loss, compute_generator_loss
from miscc.datasets import AudioSet
from miscc.utils import compute_discriminator_wgan_loss, compute_generator_wgan_loss
#from tensorboard import summary
# from tensorflow.summary import FileWriter


from tensorboardX import SummaryWriter

def create_dataset(filename):
    return AudioSet(filename)

class GANTrainer(object):
    def __init__(self, output_dir):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = SummaryWriter(self.log_dir)

        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        if cfg.CUDA:
            self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
            torch.cuda.set_device(self.gpus[0])
            cudnn.benchmark = True
        else:
            self.batch_size = cfg.TRAIN.BATCH_SIZE
            self.num_gpus = 0


    def load_network_embedding(self):
        from model import EmbeddingNet
        netE = EmbeddingNet(cfg.AUDIO.FEATURE_DIM, cfg.AUDIO.DIMENSION)
        netE.apply(weights_init)
        if cfg.EMB_NET != '':
            state_dict = torch.load(cfg.EMB_NET,
                                    map_location=lambda storage, loc: storage)
            netE.load_state_dict(state_dict)
            for p in netE.parameters():
                p.requires_grad=False
            print('Load from: ', cfg.EMB_NET)
        if cfg.CUDA:
            netE.cuda()
        return netE


    def load_network_stageI(self):
        from model import STAGE1_G, STAGE1_D
        netG = STAGE1_G()
        netG.apply(weights_init)
        netD = STAGE1_D()
        netD.apply(weights_init)

        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    # ############# For training stageII GAN  #############
    def load_network_stageII(self):
        from model import STAGE1_G, STAGE2_G_twostream, STAGE2_D

        Stage1_G = STAGE1_G()
        netG = STAGE2_G_twostream(Stage1_G)
        netG.apply(weights_init)
        print(netG)
        if cfg.NET_G != '':
            state_dict = \
                torch.load(cfg.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        elif cfg.STAGE1_G != '':
            state_dict = \
                torch.load(cfg.STAGE1_G,
                           map_location=lambda storage, loc: storage)
            netG.STAGE1_G.load_state_dict(state_dict)
            print('Load from: ', cfg.STAGE1_G)
        else:
            print("Please give the Stage1_G path")
            return

        netD = STAGE2_D()
        netD.apply(weights_init)
        if cfg.NET_D != '':
            state_dict = \
                torch.load(cfg.NET_D,
                           map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        print(netD)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def train(self, data_loader, stage=1):
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        if cfg.DATASET_NAME == 'audioset':
            # print('building')
            netE = self.load_network_embedding()
            # print('done building')
            # summary(netE, input_size=(128, 430))

        nz = cfg.Z_DIM
        wgan_d_count = 0
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = torch.FloatTensor(batch_size, nz).normal_(0, 1)
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()
            #fixed_noise_test = fixed_noise_test.cuda()
        one = torch.FloatTensor([1])
        mone = one * -1

        if cfg.CUDA:
            one = one.cuda()
            mone = mone.cuda()

        generator_lr = cfg.TRAIN.GENERATOR_LR
        discriminator_lr = cfg.TRAIN.DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN.LR_DECAY_EPOCH
        optimizerD = \
            optim.Adam(netD.parameters(),
                       lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(cfg.TRAIN.ADAM_BETA1, cfg.TRAIN.ADAM_BETA2))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(cfg.TRAIN.ADAM_BETA1, cfg.TRAIN.ADAM_BETA2))
        count = 0
        # fix data to overfit
        #for data in data_loader:
        #    break
        data_dir = ["/ext2/audioset/small", "/ext/audioset/small"]
        print(data_dir)
        #with Pool() as pool:
        #    dataset_list = pool.map(create_dataset, data_dir)
        #print("Done loading all data")
        #dataloader1 = torch.utils.data.DataLoader(dataset_list[0], batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.WORKERS)
        #dataloader2 = torch.utils.data.DataLoader(dataset_list[1], batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.WORKERS)
        dataset1 = AudioSet(data_dir[0])
        print("Done loading data set 1")
        dataset2 = AudioSet(data_dir[1])
        print("Done loading data set 2")
        dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.WORKERS)
        dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.WORKERS)
        
        for epoch in range(self.max_epoch):
            start_t = time.time()
            print("running epoch {}".format(epoch))
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr


            # reuse data ev5rytime
            for i, data in enumerate(dataloader1):
                ######################################################
                # (1) Prepare training data
                ######################################################
                if cfg.DATASET_NAME == 'audioset':
                    #print(data[0].shape)
                    #print(data[1].shape)
                    if cfg.CUDA:
                        audio_feat = data[0].to(device=torch.device('cuda'), dtype=torch.float)
                        real_imgs = data[1].to(device=torch.device('cuda'), dtype=torch.float)
                    else:
                        audio_feat = data[0].type(torch.FloatTensor)
                        real_imgs = data[1].type(torch.FloatTensor)
                    # embedding = netE(audio_feat)
                    if data[0].shape[0] != cfg.TRAIN.BATCH_SIZE:
                           print("Ignoring batch {}".format(i))
                           break
                           # ignore batch has size != Bathc size 
                    with torch.no_grad():
                        m = list(netE._modules.values())[0]
                        ft = audio_feat
                        ftlist = []
                        for j, module in enumerate(m):
                            nft = module(ft)
                            ftlist.append(nft)
                            ft = nft
                        embedding = ftlist[-2]
#     ee = torch.sum((embedding > 0).to(device=torch.device('cuda'), dtype=torch.float)*embedding)
                    #     print(ee, torch.sum(embedding))
                else:
                    real_img_cpu, embedding = data
                    real_img_cpu = real_img_cpu.type(torch.FloatTensor)
                    real_imgs = Variable(real_img_cpu)
                    embedding = embedding.type(torch.FloatTensor)
                    embedding = Variable(embedding)
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    embedding = embedding.cuda()
                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (embedding, noise)
                #inputs = embedding, fixed_noise
                if cfg.CPU:
                    _, fake_imgs, mu, logvar = netG(*inputs)
                else:
                    _, fake_imgs, mu, logvar = \
                        nn.parallel.data_parallel(netG, inputs, self.gpus)

                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()
                if cfg.TRAIN.USE_WGAN:
                    wgan_d_count += 1
                    errD, wasserstein_d, gp = compute_discriminator_wgan_loss(netD, real_imgs, fake_imgs, self.gpus, mu, cfg.WGAN.LAMBDA)
                    errD.backward()
                    skip_generator_update = (wgan_d_count % cfg.WGAN.N_D != 0)
                    for param in netD.parameters():
                        if param.requires_grad:
                            #print(param.grad)   
                           pass
                else:
                    errD, errD_real, errD_wrong, errD_fake = \
                        compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                                   real_labels, fake_labels,
                                                   mu, self.gpus, use_wrong_data=cfg.TRAIN.USE_WRONG_DATA)
                    errD.backward()
                    skip_generator_update = False
                optimizerD.step()
                ############################
                # (2) Update G network
                ###########################
                if not skip_generator_update:
                    netG.zero_grad()
                    if cfg.TRAIN.USE_WGAN:
                        errG = compute_generator_wgan_loss(netD, fake_imgs,
                                                            mu, self.gpus)
                        #errG.backward(mone)
                        #errG = -errG
                        kl_loss = cfg.TRAIN.COEFF.KL * KL_loss(mu, logvar)
                        #kl_loss.backward()
                        errG_total = -errG + kl_loss
                        errG_total.backward()
                        for param in netG.parameters():
                            if param.requires_grad:
                                pass
                #print(param.grad)   
                    else:
                        errG = compute_generator_loss(netD, fake_imgs,
                                                      real_labels, mu, self.gpus)
                        kl_loss = KL_loss(mu, logvar)
                        errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL
                        errG_total.backward()
                    optimizerG.step()
                count = count + 1
                # if i % 50 == 0:
                ###########################
                # output progress
                ###########################

                if count % 100 == 0 and count != 0:
                    if cfg.TRAIN.USE_WGAN:
                        self.summary_writer.add_scalar('GP', gp, count)
                        self.summary_writer.add_scalar('D_Loss', errD, count)
                        if count < 5:
                            errG = 0
                        self.summary_writer.add_scalar('G_loss', errG_total, count)
                        self.summary_writer.add_scalar('W_Loss', wasserstein_d,count)
                        self.summary_writer.add_scalar('KL_loss', kl_loss.data[0],count)
                        self.summary_writer.add_scalar('errG', errG, count)
                    else:
                        self.summary_writer.add_scalar('D_loss', errD.data[0],count)
                        self.summary_writer.add_scalar('D_loss_real', errD_real,count)
                        self.summary_writer.add_scalar('D_loss_real', errD_real,count)
                        self.summary_writer.add_scalar('D_loss_wrong', errD_wrong,count)
                        self.summary_writer.add_scalar('D_loss_fake', errD_fake,count)
                        self.summary_writer.add_scalar('G_loss', errG.data[0],count)
                        self.summary_writer.add_scalar('KL_loss', kl_loss.data[0],count)
                    # save the image result for each epoch
                    inputs = (embedding, fixed_noise)
                    #inputs = (embedding, fixed_noise_test)
                    if cfg.CPU:
                        lr_fake, fake, _, _ = netG(*inputs)
                    else:
                        lr_fake, fake, _, _ = \
                            nn.parallel.data_parallel(netG, inputs, self.gpus)
                    save_img_results(real_imgs, fake, epoch, self.image_dir)
                    if lr_fake is not None:
                        save_img_results(None, lr_fake, epoch, self.image_dir)
            print("Done dataset 1")   
            for i, data in enumerate(dataloader2):
                ######################################################
                # (1) Prepare training data
                ######################################################
                if cfg.DATASET_NAME == 'audioset':
                    if cfg.CUDA:
                        audio_feat = data[0].to(device=torch.device('cuda'), dtype=torch.float)
                        real_imgs = data[1].to(device=torch.device('cuda'), dtype=torch.float)
                    else:
                        audio_feat = data[0].type(torch.FloatTensor)
                        real_imgs = data[1].type(torch.FloatTensor)
                    # embedding = netE(audio_feat)
                    if data[0].shape[0] != cfg.TRAIN.BATCH_SIZE:
                           print("Ignoring batch {}".format(i))
                           break
                    with torch.no_grad():
                        m = list(netE._modules.values())[0]
                        ft = audio_feat
                        ftlist = []
                        for j, module in enumerate(m):
                            nft = module(ft)
                            ftlist.append(nft)
                            ft = nft
                        embedding = ftlist[-2]
#     ee = torch.sum((embedding > 0).to(device=torch.device('cuda'), dtype=torch.float)*embedding)
                    #     print(ee, torch.sum(embedding))
                else:
                    real_img_cpu, embedding = data
                    real_img_cpu = real_img_cpu.type(torch.FloatTensor)
                    real_imgs = Variable(real_img_cpu)
                    embedding = embedding.type(torch.FloatTensor)
                    embedding = Variable(embedding)
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    embedding = embedding.cuda()
                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                inputs = (embedding, noise)
                #inputs = embedding, fixed_noise
                if cfg.CPU:
                    _, fake_imgs, mu, logvar = netG(*inputs)
                else:
                    _, fake_imgs, mu, logvar = \
                        nn.parallel.data_parallel(netG, inputs, self.gpus)

                ############################
                # (3) Update D network
                ###########################
                netD.zero_grad()
                if cfg.TRAIN.USE_WGAN:
                    wgan_d_count += 1
                    errD, wasserstein_d, gp = compute_discriminator_wgan_loss(netD, real_imgs, fake_imgs, self.gpus, mu, cfg.WGAN.LAMBDA)
                    errD.backward()
                    skip_generator_update = (wgan_d_count % cfg.WGAN.N_D != 0)
                    for param in netD.parameters():
                        if param.requires_grad:
                            #print(param.grad)   
                           pass
                else:
                    errD, errD_real, errD_wrong, errD_fake = \
                        compute_discriminator_loss(netD, real_imgs, fake_imgs,
                                                   real_labels, fake_labels,
                                                   mu, self.gpus, use_wrong_data=cfg.TRAIN.USE_WRONG_DATA)
                    errD.backward()
                    skip_generator_update = False
                optimizerD.step()
                ############################
                # (2) Update G network
                ###########################
                if not skip_generator_update:
                    netG.zero_grad()
                    if cfg.TRAIN.USE_WGAN:
                        errG = compute_generator_wgan_loss(netD, fake_imgs,
                                                            mu, self.gpus)
                        #errG = -errG
                        kl_loss = cfg.TRAIN.COEFF.KL * KL_loss(mu, logvar)
                        #kl_loss.backward()
                        errG_total = -errG + kl_loss
                        errG_total.backward()
                    else:
                        errG = compute_generator_loss(netD, fake_imgs,
                                                      real_labels, mu, self.gpus)
                        kl_loss = KL_loss(mu, logvar)
                        errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL
                        errG_total.backward()
                    optimizerG.step()
                count = count + 1
                # if i % 50 == 0:
                ###########################
                # output progress
                ###########################

                if count % 100 == 0 and count != 0:
                    if cfg.TRAIN.USE_WGAN:
                        self.summary_writer.add_scalar('GP', gp, count)
                        self.summary_writer.add_scalar('D_Loss', errD, count)
                        if count < 5:
                            errG = 0
                        self.summary_writer.add_scalar('G_loss', errG_total, count)
                        self.summary_writer.add_scalar('KL_loss', kl_loss.data[0],count)
                        self.summary_writer.add_scalar('W_Loss', wasserstein_d,count)
                        self.summary_writer.add_scalar('errG', errG, count)
                    else:
                        self.summary_writer.add_scalar('D_loss', errD.data[0],count)
                        self.summary_writer.add_scalar('D_loss_real', errD_real,count)
                        self.summary_writer.add_scalar('D_loss_real', errD_real,count)
                        self.summary_writer.add_scalar('D_loss_wrong', errD_wrong,count)
                        self.summary_writer.add_scalar('D_loss_fake', errD_fake,count)
                        self.summary_writer.add_scalar('G_loss', errG.data[0],count)
                        self.summary_writer.add_scalar('KL_loss', kl_loss.data[0],count)
                    # save the image result for each epoch
                    inputs = (embedding, fixed_noise)
                    #inputs = (embedding, fixed_noise_test)
                    if cfg.CPU:
                        lr_fake, fake, _, _ = netG(*inputs)
                    else:
                        lr_fake, fake, _, _ = \
                            nn.parallel.data_parallel(netG, inputs, self.gpus)
                    save_img_results(real_imgs, fake, epoch, self.image_dir)
                    if lr_fake is not None:
                        save_img_results(None, lr_fake, epoch, self.image_dir)
            end_t = time.time()
            if cfg.TRAIN.USE_WGAN:
                print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f
                         wassterain_loss: %.4f
                         Total Time: %.2fsec
                      '''
                      % (epoch, self.max_epoch, i, len(dataloader1) + len(dataloader2),
                         errD, errG,
                         wasserstein_d, (end_t - start_t)))
            else:
                print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                         Loss_real: %.4f Loss_wrong:%.4f Loss_fake %.4f
                         Total Time: %.2fsec
                      '''
                      % (epoch, self.max_epoch, i, len(data_loader),
                         errD.data[0], errG.data[0], kl_loss.data[0],
                         errD_real, errD_wrong, errD_fake, (end_t - start_t)))
            if epoch % self.snapshot_interval == 0:
                save_model(netG, netD, epoch, self.model_dir)
        #
        save_model(netG, netD, self.max_epoch, self.model_dir)
        #
        self.summary_writer.close()


    def sample(self, datapath, stage=1):
        netG, _ = self.load_network_stageI()
        netG.eval()
        save_dir = 'sample_G_{}'.format(datapath)

        # Load text embeddings generated from the encoder
        #t_file = torchfile.load(datapath)
        #captions_list = t_file.raw_txt
        val_set = AudioSet(datapath, label=True)
        dataloader = torch.utils.data.DataLoader(
            val_set, batch_size=64,
            shuffle=False)
        netE = self.load_network_embedding()
        nz = cfg.Z_DIM
        noise = torch.FloatTensor(64, nz)
        sampe_result = []
        for i, data in enumerate(dataloader):
            if cfg.DATASET_NAME == 'audioset':
                if cfg.CUDA:
                    audio_feat = data[0].to(device=torch.device('cuda'), dtype=torch.float)
                    real_imgs = data[1].to(device=torch.device('cuda'), dtype=torch.float)
                    label = data[2].to(device=torch.device('cuda'), dtype=torch.long)
                else:
                    audio_feat = data[0].type(torch.FloatTensor)
                    real_imgs = data[1].type(torch.FloatTensor)
                    label = data[2].type(torch.LongTensor)
                # embedding = netE(audio_feat)
                if data[0].shape[0] != cfg.TRAIN.BATCH_SIZE:
                       print("Ignoring batch {}".format(i))
                       break
                with torch.no_grad():
                    m = list(netE._modules.values())[0]
                    ft = audio_feat
                    ftlist = []
                    for j, module in enumerate(m):
                        nft = module(ft)
                        ftlist.append(nft)
                        ft = nft
                    embedding = ftlist[-2]
            else:
                real_img_cpu, embedding = data
                real_img_cpu = real_img_cpu.type(torch.FloatTensor)
                real_imgs = Variable(real_img_cpu)
                embedding = embedding.type(torch.FloatTensor)
                embedding = Variable(embedding)
            if cfg.CUDA:
                real_imgs = real_imgs.cuda()
                embedding = embedding.cuda()
                noise.data.normal_(0, 1)
                inputs = (embedding, noise)
                if cfg.CPU:
                    _, fake_imgs, mu, logvar = netG(*inputs)
                else:
                    _, fake_imgs, mu, logvar = \
                        nn.parallel.data_parallel(netG, inputs, self.gpus)

            for k in range(len(data)):
                sampe_result.append({'real': real_imgs[k].cpu().numpy(), 'audio': audio_feat[k], 'label': label[i]})
        print("sampled {} bathc of data".format(i))
        np.save(np.array(sampe_result), save_dir + ".npz")
        print("saved {}".format(save_dir))

class EmbeddingNetTrainer(object):
    def __init__(self, cfg, output_dir=None, model=None):
        self.output_dir = output_dir
        self.feature_dim = cfg.AUDIO.FEATURE_DIM
        self.output_dim = cfg.AUDIO.DIMENSION
        self.use_gpu = cfg.CUDA
        self.epochs = cfg.TRAIN.MAX_EPOCH
        self.n_workers = int(cfg.WORKERS)
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.learning_rate = 8e-4
        self.num_classes = cfg.NUM_CLASSES
        if model is not None:
            self.model = model
        else:
            self.build_model()
        if self.use_gpu:
            self.model = self.model.to(device=torch.device('cuda'))

    def build_model(self):
        from model import EmbeddingNet
        self.embnet = EmbeddingNet(self.feature_dim, self.output_dim)
        self.model = nn.Sequential( self.embnet,
                                    nn.ReLU(),
                                    nn.Linear(self.output_dim, self.num_classes),
                                    )
    def train(self, train_set, eval_set=None):
        dataloader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size,
            shuffle=True, num_workers=self.n_workers)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_eval_acc = -1.0
        for e in range(self.epochs):
            print('Epoch %d / %d' % (e, self.epochs))
            st = time.time()
            # Train
            num_correct = 0
            num_samples = 0
            for i, data in enumerate(dataloader):
                self.model = self.model.train()  # put model to training mode
                x = data['audio']
                y = data['label']
                if self.use_gpu:
                    x = x.to(device=torch.device('cuda'), dtype=torch.float)
                    y = y.to(device=torch.device('cuda'), dtype=torch.long)

                scores = self.model(x)
                scores = scores.view(-1, self.num_classes)
                loss = torch.nn.functional.cross_entropy(scores, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    _, preds = scores.max(1)
                    num_correct += (preds == y).sum()
                    num_samples += preds.size(0)

                if i % 200 == 0:
                    print('Epoch %d, Iteration %d, loss = %.4f' % (e, i, loss.item()))

            acc = float(num_correct) / num_samples if num_samples > 0 else 0.0
            print('Epoch %d took %.2f mins' % (e, (time.time()-st)/60.0))
            print('Train: Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

            # Evaluate
            if eval_set is not None:
                acc = self.evaluate(eval_set)
                if acc >= best_eval_acc and self.output_dir is not None:
                    save_dir = os.path.join(self.output_dir, 'embnet')
                    mkdir_p(save_dir)
                    torch.save( self.embnet.state_dict(),
                                os.path.join(save_dir, 'embnet.pth')
                                )
                    print('Save Embedding Net model')
                    best_eval_acc = acc
                                

    def evaluate(self, eval_set):
        dataloader = torch.utils.data.DataLoader(
            eval_set, batch_size=self.batch_size, num_workers=self.n_workers)
        num_correct = 0
        num_samples = 0
        model = self.model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                x = data['audio']
                y = data['label']
                if self.use_gpu:
                    x = x.to(device=torch.device('cuda'), dtype=torch.float)  # move to device, e.g. GPU
                    y = y.to(device=torch.device('cuda'), dtype=torch.long)
                scores = model(x)
                scores = scores.view(-1, self.num_classes)
                loss = torch.nn.functional.cross_entropy(scores, y)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples if num_samples > 0 else 0.0
            print('Eval:  Got %d / %d correct (%.2f), loss %.4f' % (num_correct, num_samples, 100 * acc, loss.item()))
        return acc

class EmbeddingNetLSTMTrainer(EmbeddingNetTrainer):
    def build_model(self):
        from model import EmbeddingNetLSTM
        self.model = nn.Sequential( EmbeddingNetLSTM(self.feature_dim, self.output_dim),
                                    nn.ReLU(),
                                    nn.Linear(self.output_dim, self.num_classes),
                                    )

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
