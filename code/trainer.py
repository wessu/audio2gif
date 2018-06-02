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

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import weights_init
from miscc.utils import save_img_results, save_model
from miscc.utils import KL_loss
from miscc.utils import compute_discriminator_loss, compute_generator_loss
from miscc.utils import compute_discriminator_wgan_loss, compute_generator_wgan_loss

from torchsummary import summary
from tensorboardX import SummaryWriter

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
                       lr=cfg.TRAIN.DISCRIMINATOR_LR, betas=(0.5, 0.999))
        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para,
                                lr=cfg.TRAIN.GENERATOR_LR,
                                betas=(0.5, 0.999))
        count = 0
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

            for i, data in enumerate(data_loader, 0):
                ######################################################
                # (1) Prepare training data
                ######################################################
                if cfg.DATASET_NAME == 'audioset':
                    if cfg.CUDA:
                        audio_feat = data['audio'].to(device=torch.device('cuda'), dtype=torch.float)
                        real_imgs = data['video'].to(device=torch.device('cuda'), dtype=torch.float)
                    else:
                        audio_feat = data['audio'].type(torch.FloatTensor)
                        real_imgs = data['video'].type(torch.FloatTensor)
                    embedding = netE(audio_feat) #
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
                    errD, wasserstein_d = compute_discriminator_wgan_loss(netD, real_imgs, fake_imgs, self.gpus, mu, cfg.WGAN.LAMBDA)
                    errD.backward()
                    skip_generator_update = (wgan_d_count % cfg.WGAN.N_D != 0)
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
                        errG.backward(mone)
                        errG = -errG
                    else:
                        errG = compute_generator_loss(netD, fake_imgs,
                                                      real_labels, mu, self.gpus)
                        kl_loss = KL_loss(mu, logvar)
                        errG_total = errG + kl_loss * cfg.TRAIN.COEFF.KL
                        errG_total.backward()
                    optimizerG.step()
                count = count + 1
                ###########################
                # output progress
                ###########################

                if i % 100 == 0:
                    if cfg.TRAIN.USE_WGAN:
                        self.summary_writer.add_scalar('D_Loss', errD, count)
                        if i != 0:
                            self.summary_writer.add_scalar('G_loss', errG, count)
                        self.summary_writer.add_scalar('W_Loss', wasserstein_d,count)
                    else:
                        self.summary_writer.add_scaler('D_loss', errD.data[0],count)
                        self.summary_writer.add_scalar('D_loss_real', errD_real,count)
                        self.summary_writer.add_scalar('D_loss_real', errD_real,count)
                        self.summary_writer.add_scalar('D_loss_wrong', errD_wrong,count)
                        self.summary_writer.add_scalar('D_loss_fake', errD_fake,count)
                        self.summary_writer.add_scalar('G_loss', errG.data[0],count)
                        self.summary_writer.add_scalar('KL_loss', kl_loss.data[0],count)
                    # save the image result for each epoch
                    inputs = (embedding, fixed_noise)
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
                      % (epoch, self.max_epoch, i, len(data_loader),
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
        #if stage == 1:
        netG, _ = self.load_network_stageI()
        #else:
        #    netG, _ = self.load_network_stageII()
        #netG.eval()

        # Load text embeddings generated from the encoder
        t_file = torchfile.load(datapath)
        captions_list = t_file.raw_txt
        embeddings = np.concatenate(t_file.fea_txt, axis=0)
        num_embeddings = len(captions_list)
        print('Successfully load sentences from: ', datapath)
        print('Total number of sentences:', num_embeddings)
        print('num_embeddings:', num_embeddings, embeddings.shape)
        # path to save generated samples
        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')]
        mkdir_p(save_dir)

        batch_size = np.minimum(num_embeddings, self.batch_size)
        nz = cfg.Z_DIM
        noise = Variable(torch.FloatTensor(batch_size, nz))
        if cfg.CUDA:
            noise = noise.cuda()
        count = 0
        while count < num_embeddings:
            if count > 3000:
                break
            iend = count + batch_size
            if iend > num_embeddings:
                iend = num_embeddings
                count = num_embeddings - batch_size
            embeddings_batch = embeddings[count:iend]
            # captions_batch = captions_list[count:iend]
            embedding = Variable(torch.FloatTensor(embeddings_batch))
            if cfg.CUDA:
                embedding = embedding.cuda()

            #######################################################
            # (2) Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            inputs = (embedding, noise)
            if cfg.CPU:
                netG(embedding, noise)
            else:
                _, fake_imgs, mu, logvar = \
                    nn.parallel.data_parallel(netG, inputs, self.gpus)
            for i in range(batch_size):
                save_name = '%s/%d.png' % (save_dir, count + i)
                im = fake_imgs[i].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                # print('im', im.shape)
                im = np.transpose(im, (1, 2, 0))
                # print('im', im.shape)
                im = Image.fromarray(im)
                im.save(save_name)
            count += batch_size

class EmbeddingNetTrainer(object):
    def __init__(self, cfg, output_dir=None, model=None):
        self.output_dir = output_dir
        self.feature_dim = cfg.AUDIO.FEATURE_DIM
        self.output_dim = cfg.AUDIO.DIMENSION
        self.use_gpu = cfg.CUDA
        self.epochs = cfg.TRAIN.MAX_EPOCH
        self.n_workers = int(cfg.WORKERS)
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.learning_rate = 1e-4
        self.num_classes = cfg.NUM_CLASSES
        if model is not None:
            self.model = model
        else:
            self.build_model()
        if self.use_gpu:
            self.model = self.model.to(device=torch.device('cuda'))

    def build_model(self):
        from model import EmbeddingNet
        self.model = nn.Sequential( EmbeddingNet(self.feature_dim, self.output_dim),
                                    nn.Linear(self.output_dim, self.num_classes)
                                    )
    def train(self, train_set, eval_set=None):
        dataloader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size,
            shuffle=True, num_workers=self.n_workers)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_eval_acc = -1.0
        for e in range(self.epochs):
            print('Epoch %d / %d' % (e, self.epochs))
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
            print('Epoch %d' % (e))
            print('Train: Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

            # Evaluate
            if eval_set is not None:
                acc = self.evaluate(eval_set)
                if acc >= best_eval_acc and self.output_dir is not None:
                    save_dir = os.path.join(self.output_dir, 'embnet')
                    mkdir_p(save_dir)
                    torch.save( self.model.state_dict(), 
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
                                    nn.Linear(self.output_dim, self.num_classes)
                                    )
