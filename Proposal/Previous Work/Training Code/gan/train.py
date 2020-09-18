import sys
import torch
import numpy as np
import scipy as sc
from scipy import signal
from torchvision import transforms, datasets
from torchvision.utils import save_image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from skimage import transform as tf
from rnn_audio import RNN
from image_encoder import Image_Encoder
import os
import time
import face_alignment
from audio_encoder import Audio_Encoder
from img_generator import UnetBlock
from discriminator import FrameDiscriminator, SequenceDiscriminator
from dataloader import get_data
from config import (SEQ_LEN, AUDIO_OUTPUT, BATCH, HIDDEN_SIZE_AUDIO, NUM_LAYERS_AUDIO, NOISE_OUTPUT, learning_rate)
from config import AUDIO_DATA_PATH, VIDEO_DATA_PATH
from PIL import Image
from pydub import AudioSegment
from pydub.utils import mediainfo
import skvideo.io  
from loss_functions import dis_loss, fdis_loss, sdis_loss, gen_loss
#import matplotlib.pyplot as plt
#import cv2
import random
import time
#sys.exit()

conversion_dict = {'s16': np.int16, 's32': np.int32}

def train_model(audios, videos, unet, frame_discriminator, sequence_discriminator, encoder, encoder_id, model_dict):
    
    img_size = model_dict["img_size"]
    rnn_gen_dim = model_dict['rnn_gen_dim']
    id_enc_dim = model_dict['id_enc_dim']
    aud_enc_dim = model_dict['aud_enc_dim']
    aux_latent = model_dict['aux_latent']
    audio_feat_len = model_dict['audio_feat_len']
    audio_feat_samples = model_dict['audio_feat_samples']
    sequential_noise = model_dict['sequential_noise']
    audio_rate = model_dict["audio_rate"]
    mean_face = model_dict["mean_face"]
    video_rate = model_dict["video_rate"]
    print(audio_feat_samples)

    # torch.autograd.set_detect_anomaly(True)

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    optimizer_unet = torch.optim.Adam(unet.parameters(), lr=0.0008)
    optimizer_fd = torch.optim.Adam(frame_discriminator.parameters(), lr=0.001)
    optimizer_sd = torch.optim.Adam(sequence_discriminator.parameters(), lr=0.001)
    
    tick = time.time()
    num_epochs = 150
    # best_loss = 1000
    best_loss = model_dict["best_loss"]
    for epoch in range(num_epochs):
        batch_g_loss = 0
        batch_d_loss = 0
        rejected = 0
        for i in range(len(videos)):
            # try:
              video_d = skvideo.io.vread(os.path.join(VIDEO_DATA_PATH,videos[i]+'.mp4'))
              # print("YES LOADED")
              # print(video_d.shape)
              video_d = video_d.transpose(0, 3, 1, 2)
              
              for j in range(video_d.shape[0]):
                  video_d[j] = (video_d[j] - video_d[j].min())/(video_d[j].max() - video_d[j].min())
              video_d = torch.from_numpy(video_d)

              #print(video_d.shape)
              video_data = Variable(video_d)  # this needs to be array of still frames

              # audio_d = np.load(os.path.join(AUDIO_DATA_PATH,audios[i]+'.wav'))
              # audio_d = torch.from_numpy(audio_d)
              # audio_d = audio_d.view(audio_d.size()[0], audio_d.size()[2], audio_d.size()[1])
              # audio_data = Variable(audio_d)
              audio = AUDIO_DATA_PATH+"/"+str(audios[i])+'.wav'
              if isinstance(audio, str):  # if we have a path then grab the audio clip
                  info = mediainfo(audio)
                  fs = int(info['sample_rate'])
                  audio = np.array(AudioSegment.from_file(audio, info['format_name']).set_channels(1).get_array_of_samples())

              if info['sample_fmt'] in conversion_dict:
                  audio = audio.astype(conversion_dict[info['sample_fmt']])
              else:
                  if max(audio) > np.iinfo(np.int16).max:
                      audio = audio.astype(np.int32)
                  else:
                      audio = audio.astype(np.int16)

              if fs is None:
                  raise AttributeError("Audio provided without specifying the rate. Specify rate or use audio file!")

              if audio.ndim > 1 and audio.shape[1] > 1:
                  audio = audio[:, 0]

              max_value = np.iinfo(audio.dtype).max
              # print(audio)
              speech = None
              if fs != audio_rate:
                  seq_length = audio.shape[0]
                  speech = torch.from_numpy(
                      signal.resample(audio, int(seq_length * audio_rate / float(fs))) / float(max_value)).float()
                  speech = speech.view(-1, 1)
              else:
                  audio = torch.from_numpy(audio / float(max_value)).float()
                  speech = audio.view(-1, 1)
              
              audio_data = speech
              
              still_frame = video_d[2]

              img_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size[0], img_size[1])),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
              # frame = preprocess_img(still_frame)
              frame = img_transform(still_frame)
              
              # frame = transforms.ToPILImage(still_frame)
              # frame = transforms.Resize((frame.shape[0], frame.shape[1]))
              # frame = transforms.ToTensor(frame)
              # frame = transforms.Normalize(frame, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).to(self.device)

              cutting_stride = int(audio_rate / float(video_rate))
              audio_seq_padding = audio_feat_samples - cutting_stride

              # Create new sequences of the audio windows
              audio_feat_seq = cut_sequence_(speech, cutting_stride, audio_seq_padding, audio_feat_samples)
              frame = frame.unsqueeze(0)
              audio_feat_seq = audio_feat_seq.unsqueeze(0)
              audio_feat_seq_length = audio_feat_seq.size()[1]

              # print(audio_feat_seq.shape)
              # print(audio_feat_seq_length)
              z = encoder(audio_feat_seq, [audio_feat_seq_length])
              
              noise = torch.FloatTensor(1, audio_feat_seq_length, aux_latent).normal_(0, 0.33)

              z_id, skips = encoder_id(frame, retain_intermediate=True)
              skip_connections = []
              for skip_variable in skips:
                  skip_connections.append(broadcast_elements_(skip_variable, z.size()[1]))
              skip_connections.reverse()

              z_id = broadcast_elements_(z_id, z.size()[1])
              
              gen_frames = unet(z, c=z_id, aux=noise, skip=skip_connections)

              optimizer_unet.zero_grad()
              optimizer_fd.zero_grad()
              optimizer_sd.zero_grad()

  
              Lambda = 100
              
              if(len(video_data)>len(gen_frames)):
                video_data = video_data[:len(gen_frames),:,:,:]
              elif(len(gen_frames)>len(video_data)):
                gen_frames = gen_frames[:len(video_data),:,:,:]
                  
              l1_loss = torch.mean(torch.mean(torch.mean(torch.mean(torch.abs(video_data - gen_frames), 1), 1), 1))
              
              temp = []
              for i in range(len(gen_frames)):
                temp.append(still_frame.numpy()[:, :, :])
              
              still_frame = torch.from_numpy(np.asarray(temp)).type(torch.FloatTensor)
              video_data = video_data.numpy()[:,:,:,:]
              video_data = torch.from_numpy(video_data).type(torch.FloatTensor)
              
              out2 = frame_discriminator(gen_frames, still_frame)
              out1 = frame_discriminator(video_data, still_frame)
              
              out3 = sequence_discriminator(video_data, audio_feat_seq, audio_feat_seq_length, model_dict)
              out4 = sequence_discriminator(gen_frames, audio_feat_seq, audio_feat_seq_length, model_dict)
              

              d_loss = -dis_loss(out2, out1, out4 ,out3)
              
              g_loss = -gen_loss(out2, out4) + Lambda*l1_loss
              g_loss.backward(retain_graph=True)
              d_loss.backward()

              optimizer_unet.step()

              optimizer_fd.step()
              optimizer_sd.step()
              print("d loss:", d_loss.data, " g loss:", g_loss.data)

              
              batch_g_loss += g_loss.data
              batch_d_loss += d_loss.data
              tock = time.time()
                
        average_batch_g_loss = batch_g_loss/(len(videos) - rejected)
        average_batch_d_loss = batch_d_loss/(len(videos) - rejected)

        if epoch%10 == 0:
            torch.save(unet, 'check.pt')

        # generate_test_images(epoch, unet)
        if average_batch_g_loss < best_loss:
            torch.save(unet, 'unet.pt')
            torch.save(frame_discriminator, view+'FrameDiscriminator.pt')
            torch.save(sequence_discriminator, view+'SequenceDiscriminator.pt')
            state = {
               'unet': unet.state_dict(),
               'frame_discriminator': frame_discriminator.state_dict(),
               'sequence_discriminator': sequence_discriminator.state_dict(),
               'optimizer_unet': optimizer_unet.state_dict(),
               'optimizer_fd': optimizer_fd.state_dict(),
               'optimizer_sd': optimizer_sd.state_dict(),
               'best_loss': best_loss,
               'img_size': img_size,
               'rnn_gen_dim': rnn_gen_dim,
               'id_enc_dim': id_enc_dim,
               'aud_enc_dim': aud_enc_dim,
               'aux_latent': aux_latent,
               'audio_feat_len': audio_feat_len,
               'audio_feat_samples': audio_feat_samples,
               'sequential_noise': sequential_noise,
               'audio_rate': audio_rate,
               'mean_face' : mean_face,
               'video_rate': video_rate
            }
            torch.save(state, 'grid.dat')

            best_loss = average_batch_g_loss
        print("E: {} G Loss: {} D loss {} R files: {}".format(epoch, average_batch_g_loss, average_batch_d_loss, rejected))


def preprocess_img(img, mean_face):
        stablePntsIDs = [33, 36, 39, 42, 45]
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device="cuda:" + str(0),
                                                   flip_input=False)
        src = fa.get_landmarks(img)[0][stablePntsIDs, :]
        dst = mean_face[stablePntsIDs, :]
        tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
        warped = tf.warp(img, inverse_map=tform.inverse, output_shape=(480,720))  # wrap the frame image
        warped = warped * 255  # note output from wrap is double image (value range [0,1])
        warped = warped.astype('uint8')

        return warped

def cut_sequence_(seq, cutting_stride, pad_samples, audio_feat_samples):
        pad_left = torch.zeros(pad_samples // 2, 1)
        pad_right = torch.zeros(pad_samples - pad_samples // 2, 1)

        seq = torch.cat((pad_left, seq), 0)
        seq = torch.cat((seq, pad_right), 0)

        stacked = seq.narrow(0, 0, audio_feat_samples).unsqueeze(0)
        iterations = (seq.size()[0] - audio_feat_samples) // cutting_stride + 1
        for i in range(1, iterations):
            stacked = torch.cat((stacked, seq.narrow(0, i * cutting_stride, audio_feat_samples).unsqueeze(0)))
        return stacked


def broadcast_elements_(batch, repeat_no):
        total_tensors = []
        for i in range(0, batch.size()[0]):
            total_tensors += [torch.stack(repeat_no * [batch[i]])]

        return torch.stack(total_tensors)
