import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable

def conv3_1d(in_planes, out_planes, stride=1, padding=False):
    "3 convolution without padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=(padding*1), bias=True)

def conv3_1dxn(in_planes, out_planes, n_layers, stride=1, padding=False):
    in_planes = [in_planes] + [out_planes] * (n_layers - 1)
    seqs = [nn.Sequential(
        conv3_1d(in_plane, out_planes, stride, padding),
        nn.BatchNorm1d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
        ) for in_plane in in_planes] + [nn.Sequential(nn.Dropout(p=0.4))]
    return nn.Sequential(*seqs)

def conv3x3_2d(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3_3d(in_vol, out_vol, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(in_vol, out_vol, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock_2d(in_planes, out_planes, scale_factor=2):
    block = nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode='nearest'),
        conv3x3_2d(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block

# Upsale the spatial size by a factor of 2
def upBlock_3d(in_vol, out_vol, scale_factor=(2,2,2)):
    block = nn.Sequential(
        nn.Upsample(scale_factor=scale_factor, mode='trilinear'),
        conv3x3_3d(in_vol, out_vol),
        nn.BatchNorm3d(out_vol),
        nn.ReLU(True))
    return block

class Squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)  # "flatten" the C * H * W values into a single vector per image

class ResBlock1d(nn.Module):
    def __init__(self, in_dim, feat_dim, stride=2, dropout=0.4):
        super(ResBlock1d, self).__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.stride = stride
        self.dropout = dropout
        self.define_module()

    def define_module(self):
        self.cell = nn.Sequential(
            # nn.Conv1d(in_dim, feat_dim, kernel_size=1, stride=1),
            # nn.BatchNorm1d(feat_dim),
            # nn.ReLU(True),
            conv3_1d(self.in_dim, self.feat_dim, stride=self.stride, padding=True),
            nn.BatchNorm1d(self.feat_dim),
            nn.ReLU(True),
            conv3_1d(self.feat_dim, self.feat_dim, stride=1, padding=True),
            nn.BatchNorm1d(self.feat_dim),
            # nn.ReLU(True),
            # nn.Conv1d(feat_dim, in_dim, kernel_size=1, stride=1),
            # nn.BatchNorm1d(feat_dim),
            )
        self.residual = nn.Sequential(
            nn.Conv1d(self.in_dim, self.feat_dim, kernel_size=1, stride=1),
            nn.BatchNorm1d(self.feat_dim),
            nn.AvgPool1d(self.stride),
            )

    def forward(self, x):
        if (self.in_dim == self.feat_dim) and (self.stride == 1):
            out = torch.nn.functional.relu((self.cell(x) + x), True)
        else:
            out = torch.nn.functional.relu((self.cell(x) + self.residual(x)), True)
        if self.dropout > 0:
            out = torch.nn.functional.dropout(out, p=self.dropout)
        return out

class EmbeddingNet(nn.Module):
    def __init__(self, feat_dim, out_dim):
        super(EmbeddingNet, self).__init__()
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.define_module()
        
    def define_module(self):
        self.embedding_net_1 = nn.Sequential(
            conv3_1dxn(self.feat_dim, 256, 5), # 420
            nn.MaxPool1d(2), # 210
            conv3_1dxn(256, 512, 5), # 200
            nn.MaxPool1d(2), # 100
            # conv3_1d(512, 1024, stride=2, padding=True), # 50
            # nn.BatchNorm1d(1024),
            # nn.ReLU(True),
            # nn.Dropout(p=0.4),
            # conv3_1d(1024, 1024, stride=2, padding=True), # 25
            # nn.BatchNorm1d(1024),
            # nn.ReLU(True),
            # nn.Dropout(p=0.4),
            ResBlock1d(512, 512, 1), # 100
            nn.MaxPool1d(2), # 50
            ResBlock1d(512, 512, 1), # 50
            nn.MaxPool1d(2), # 25
            ResBlock1d(512, 1024, 1), # 25
            nn.MaxPool1d(2), # 12
            ResBlock1d(1024, 1024), # 12
            nn.MaxPool1d(6), # 2
            # ResBlock1d(1024, 1024), # 3
            # ResBlock1d(1024, 1024, 3), # 1
            # conv3_1d(1024, 1024, stride=1, padding=True), # 12
            # nn.BatchNorm1d(1024),
            # nn.ReLU(True),
            # nn.MaxPool1d(12),
            Squeeze(), # shape (N, 2048)
            nn.Linear(1024, self.out_dim)
            )
        # self.lstm = nn.LSTM(1024, 1024)
        # self.embedding_net_2 = nn.Sequential(
            # Flatten(), # (N, 1024 * 5)
            # nn.Linear(1024 * 5, 128 * 5)
            # )

        # self.e3 = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(128*430, 1024)
        #     )

    def forward(self, features):
        output = self.embedding_net_1(features)
        return output
        # output = torch.transpose(output, 1, 2)
        # output = torch.transpose(output, 0, 1)

        # output, (h,c) = self.lstm(output)
        # output = torch.transpose(output, 0, 2)
        # output = torch.transpose(output, 0, 1)
        # return self.embedding_net_2(output)
        # return self.e3(features)

class EmbeddingNetLSTM(nn.Module):
    def __init__(self, feat_dim, out_dim):
        super(EmbeddingNetLSTM, self).__init__()
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.define_module()
        
    def define_module(self):
        self.embedding_net_1 = nn.Sequential(
            conv3_1dxn(self.feat_dim, 256, 5),
            nn.MaxPool1d(2),
            conv3_1dxn(256, 512, 5),
            nn.MaxPool1d(2),
            conv3_1d(512, 1024, stride=2, padding=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            conv3_1d(1024, 1024, stride=2, padding=True), # 25
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(p=0.4),
            )
        self.lstm = nn.LSTM(1024, 512)
        self.embedding_net_2 = nn.Sequential(
            nn.MaxPool1d(5),
            Flatten(), # (N, 1024 * 5)
            nn.Linear(512 * 5, self.out_dim),
            )

        # self.e3 = nn.Sequential(
        #     Flatten(),
        #     nn.Linear(128*430, 1024)
        #     )

    def forward(self, features):
        output = self.embedding_net_1(features)
        output = torch.transpose(output, 1, 2)
        output = torch.transpose(output, 0, 1)

        output, (h,c) = self.lstm(output)
        output = torch.transpose(output, 0, 1)
        output = torch.transpose(output, 1, 2)
        return self.embedding_net_2(output)
        # return self.e3(features)

class Flatten(nn.Module):
    def forward(self, x):
        N, F, T = x.size()
        return x.view(N, -1)

class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        # self.t_dim = cfg.AUDIO.DIMENSION
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, audio_embedding): # audio encoding
        x = self.relu(self.fc(audio_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, audio_embedding):
        mu, logvar = self.encode(audio_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
                conv3x3_2d(ndf * 8 + nef, ndf * 8),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        # conditioning output
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            print("c_code shape {}".format(c_code.shape))
            print("h_code shape {}".format(h_code.shape))
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)

class STAGE1_G(nn.Module):
    def __init__(self):
        super(STAGE1_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM * 8
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.z_dim = cfg.Z_DIM
        self.define_module()

    def define_module(self):
        ninput = self.z_dim + self.ef_dim
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock_2d(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock_2d(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock_2d(ngf // 4, ngf // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock_2d(ngf // 8, ngf // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3_2d(ngf // 16, 3),
            nn.Tanh())

    def forward(self, text_embedding, noise):
        c_code, mu, logvar = self.ca_net(text_embedding)
        z_c_code = torch.cat((noise, c_code), 1)
        h_code = self.fc(z_c_code)

        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 3 x 64 x 64
        fake_img = self.img(h_code)
        return None, fake_img, mu, logvar

class STAGE1_D(nn.Module):
    def __init__(self):
        super(STAGE1_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False), # 3 x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            # state size (ndf * 8) x 4 x 4)
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        self.get_uncond_logits = None

    def forward(self, image):
        img_embedding = self.encode_img(image)
        return img_embedding

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3_2d(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3_2d(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out




class STAGE2_D(nn.Module):
    def __init__(self):
        super(STAGE2_D, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.define_module()

    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        # img size = 4 * 256 * 256
        self.encode_img = nn.Sequential(
            nn.Conv3d(3, ndf, 4, stride=2, padding=1, bias=False),  # 2 * 128 * 128 * ndf
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False), # 1 * 64 * 64 * ndf
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # 64 * 64 * ndf * 2
            Squeeze(), # 64 * 64 * ndf * 2
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # 32 * 32 * ndf * 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # 16 * 16 * ndf * 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 8 * 8 * ndf * 16
            nn.Conv2d(ndf * 16, ndf * 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 32
            conv3x3_2d(ndf * 32, ndf * 16),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),  # 4 * 4 * ndf * 16
            conv3x3_2d(ndf * 16, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)  # 4 * 4 * ndf * 8
        )
        self.get_cond_logits = D_GET_LOGITS(ndf, nef, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf, nef, bcondition=False)

    def forward(self, image):
        print("image shape = {}".format(image.shape))
        img_embedding = self.encode_img(image)
        print("img_embedding shape = {}".format(img_embedding.shape))
        return img_embedding

class STAGE2_G_twostream(nn.Module):
    def __init__(self, STAGE1_G):
        super(STAGE2_G_twostream, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM #c_dim
        self.z_dim = cfg.Z_DIM
        self.STAGE1_G = STAGE1_G
        # fix parameters of stageI GAN
        for param in self.STAGE1_G.parameters():
            param.requires_grad = False
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        self.ca_net = CA_NET() # c_dim

        # --> 4ngf x 16 x 16
        self.encoder = nn.Sequential(
            conv3x3_2d(3, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))

        self.hr_joint = nn.Sequential(
            conv3x3_2d(self.ef_dim + ngf * 4, ngf * 4),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))

        self.residual = self._make_layer(ResBlock, ngf * 4)

        # upsample background
        self.upsample1 = upBlock_2d(ngf * 4, ngf * 2)
        # --> ngf x 64 x 64
        self.upsample2 = upBlock_2d(ngf * 2, ngf)
        # --> ngf // 2 x 128 x 128
        self.upsample3 = upBlock_2d(ngf, ngf // 2)
        # --> ngf // 4 x 256 x 256
        self.upsample4 = upBlock_2d(ngf // 2, ngf // 4)
        # --> 3 x 256 x 256
        self.background = nn.Sequential(
            conv3x3_2d(ngf // 4, 3),
            nn.Tanh())

        # upsample foreground
        self.upsample1_3d = upBlock_3d(ngf * 4, ngf * 2, scale_factor=(1,2,2))
        # --> ngf x 1 x 64 x 64
        self.upsample2_3d = upBlock_3d(ngf * 2, ngf, scale_factor=(1,2,2))
        # --> ngf // 2 x 1 x 128 x 128
        self.upsample3_3d = upBlock_3d(ngf, ngf // 2)
        # --> ngf // 4 x 2 x 256 x 256
        self.upsample4_3d = upBlock_3d(ngf // 2, ngf // 4)
        # --> 3 x x 4 x 256 x 256
        self.foreground = nn.Sequential(
            conv3x3_3d(ngf // 4, 3),
            nn.Tanh())

        # forground_background_mask
        self.foreground_mask = nn.Sequential(
            conv3x3_3d(ngf// 4, 1),
            nn.Tanh())

    def forward(self, audio_embedding, noise):
        _, stage1_img, _, _ = self.STAGE1_G(audio_embedding, noise)
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img)

        c_code, mu, logvar = self.ca_net(audio_embedding)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 16, 16)
        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code)

        h_code_2d = self.upsample1(h_code)
        h_code_2d = self.upsample2(h_code_2d)
        h_code_2d = self.upsample3(h_code_2d)
        h_code_2d = self.upsample4(h_code_2d)

        background = self.background(h_code_2d)
        print("h_code shape = {}".format(h_code.shape))
        print("h_code_2d shape = {}".format(h_code_2d.shape))
        h_code_3d = torch.unsqueeze(h_code, 2)
        print("unsqueezed shape = {}".format(h_code_3d.shape))
        h_code_3d = self.upsample1_3d(h_code_3d)
        h_code_3d = self.upsample2_3d(h_code_3d)
        h_code_3d = self.upsample3_3d(h_code_3d)
        h_code_3d = self.upsample4_3d(h_code_3d)

        foreground = self.foreground(h_code_3d)
        print("foreground shape = {}".format(foreground.shape))
        mask = self.foreground_mask(h_code_3d)
        expanded_background = torch.unsqueeze(background, 2)
        foreground_D = mask.shape[2]
        expanded_background = expanded_background.repeat(1,1,foreground_D, 1,1)
        print("expanded background shape = {}".format(expanded_background.shape))
        fake_img = mask * expanded_background + (1-mask) * foreground

        return stage1_img, fake_img, mu, logvar


class BASELINE_GIF_G(nn.Module):
    def __init__(self, BASELINE_GIF_G):
        super(BASELINE_GIF_G, self).__init__()