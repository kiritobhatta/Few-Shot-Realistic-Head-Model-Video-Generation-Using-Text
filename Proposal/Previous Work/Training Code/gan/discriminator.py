import torch
import numpy as np
import scipy as sc
from torchvision import transforms, datasets
from rnn_audio import RNN
from image_encoder import Image_Encoder
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class Convolution(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride):
        super(Convolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel,
                stride=stride),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class FrameDiscriminator(nn.Module):
    def __init__(self):
        super(FrameDiscriminator, self).__init__()

        self.c1 = Convolution(6, 64, 4, 2)
        self.c2 = Convolution(64, 128, 4, 2)
        self.c3 = Convolution(128, 256, 4, 2)
        self.c4 = Convolution(256, 512, 4, 2)
        self.c5 = Convolution(512, 1024, 4, 2)
        self.c6 = nn.Linear(1024, 128)
        self.c7 = nn.Linear(128, 1)

    def forward(self, target, still_image):
        x = torch.cat((target, still_image), dim=1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.c5(x)
        x = x.view(-1, 1024)
        # print("x size ", x.size())
        x = self.c6(x)
        out = self.c7(x)
        out = torch.sigmoid(out)
        return out


class SequenceDiscriminator(nn.Module):
    def __init__(self):
        super(SequenceDiscriminator, self).__init__()
        # self.audio_encoder = Audio_Encoder()
        # self.image_encoder = Image_Encoder()
        self.gru_image = nn.GRU(128, 50, 2)
        self.gru_audio = nn.GRU(256, 50, 2)
        self.fc1 = nn.Linear(50, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, i_i, audio_feat_seq, audio_feat_seq_length, model_dict):
        img_size = model_dict["img_size"]
        rnn_gen_dim = model_dict['rnn_gen_dim']
        id_enc_dim = model_dict['id_enc_dim']
        aud_enc_dim = model_dict['aud_enc_dim']
        aux_latent = model_dict['aux_latent']
        audio_feat_len = model_dict['audio_feat_len']
        audio_feat_samples = model_dict['audio_feat_samples']
        sequential_noise = model_dict['sequential_noise']
        audio_rate = model_dict["audio_rate"]
        # print("sd")
        
        # print(i_i.shape)
        imgenco = Image_Encoder(id_enc_dim, img_size)
        o_i = imgenco(i_i)
        # print("o_i ", o_i.size())
        
        o_i = o_i.view(o_i.size()[0], 1, o_i.size()[1])
        # print(o_i.shape)
        o_i, h_n = self.gru_image(o_i)
        o_i = o_i.transpose(1, 0)
        # print(o_i.shape)
        o_i = F.interpolate(o_i, 50, mode='linear')
        # print(o_i.shape)
        # print("o_i after GRU", o_i.size())
        # print("Audio Encoder")
        audioenco = RNN(audio_feat_len, aud_enc_dim, rnn_gen_dim,
                           audio_rate, init_kernel=0.005, init_stride=0.001)
        o_a = audioenco(audio_feat_seq, [audio_feat_seq_length])
        # print("o_a ", o_a.size())
        # o_a = o_a.view(o_a.size()[0], 1, o_a.size())
        # print(o_a.shape)
        # o_a = o_a.view(o_a.size()[0], 1, o_a.size())
        o_a, h_n = self.gru_audio(o_a)
        # print(o_a.shape)
        # print("o_a after GRU", o_a.size())
        x = torch.cat([o_a, o_i], dim=1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        out = torch.sigmoid(x)
        return out