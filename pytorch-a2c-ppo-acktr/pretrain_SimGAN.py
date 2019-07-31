import copy
import glob
import os
import time
import types
from collections import deque

#import gym
#import gym_miniworld
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#import algo
from arguments import get_args
#from envs import make_vec_envs
from model import VAE, VAER, SimGAN
from storage import RolloutStorage
#from visualize import visdom_plot
import argparse
from utils import make_var
import math
#from torch.autograd import Variable

class trainSimGAN():
    def __init__(self, device, model, lr, eps, input_train, y_train, input_eval, y_eval, model_path, batch_size = 32):
        self.model = model.to(device)
        self.optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr, eps=eps)
        self.optimizer_G = optim.Adam(model.decoder.parameters(), lr=lr, eps=eps)
        self.optimizer_E = optim.Adam(model.encoder.parameters(), lr=lr, eps=eps)
        self.batch_size = batch_size
        self.input_train = input_train
        self.y_train = y_train
        self.input_eval = input_eval
        self.y_eval = y_eval
        self.BCELoss = nn.BCELoss(reduction='mean')
        self.adversarial_loss = nn.BCELoss(reduction='mean')
        self.MSELoss = torch.nn.MSELoss()
        self.CELoss = torch.nn.CrossEntropyLoss()
        self.model_path = model_path
        self.real_label = make_var(torch.Tensor(self.batch_size, 1).fill_(1.0))
        self.fake_label = make_var(torch.Tensor(self.batch_size, 1).fill_(0.0))
        
        
    def train(self, dataset=None):
        if dataset == None:
            data = self.input_train
            y = self.y_train
        else:
            raise NotImplementedError
        self.model.train()
        for e in range(30):
            num_batch = len(data)//self.batch_size
            #idx = 0
            for i in range(num_batch):
                batch = data[i*self.batch_size:(i+1)*self.batch_size]
                batch_y = y[i*self.batch_size:(i+1)*self.batch_size]
                batch = make_var(batch)
                batch_y = make_var(batch_y)
                

                #fake
                for j in range(8):
                    self.optimizer_G.zero_grad()
                    noise = torch.randn(self.batch_size, 82)
                    noise = make_var(noise)
                    noise_with_c = torch.cat((noise,batch_y.view(self.batch_size,18)), dim=1)
                    decoded_fake = self.model.decode(noise_with_c)
                    dis_input_fake = self.model.prepare_dis_input(decoded_fake, batch_y.view(self.batch_size,18))
                    D_fake = self.model.discriminator(dis_input_fake)
                    G_loss = self.adversarial_loss(D_fake,self.real_label)
                    G_loss.backward()
                    self.optimizer_G.step()
                
                
                # real
                self.optimizer_D.zero_grad()
                dis_input_real = self.model.prepare_dis_input(batch, batch_y.view(self.batch_size,18))
                D_real = self.model.discriminator(dis_input_real)
                D_real_loss = self.adversarial_loss(D_real,self.real_label)
                
                D_fake = self.model.discriminator(dis_input_fake.detach())
                D_fake_loss = self.adversarial_loss(D_fake,self.fake_label)
                D_loss = (D_real_loss + D_fake_loss)/2
                D_loss.backward()
                self.optimizer_D.step()

                # supervised for c category
                self.optimizer_E.zero_grad()
                encoder_output = self.model.encoder(batch)
                
                latent_c = encoder_output[:,82:].view(self.batch_size, 6, 5)
                c_continuous = latent_c[:,:,3:]
                c_category_logits = latent_c[:,:,:3]
                c_category_logits_reshaped = c_category_logits.contiguous().view(-1,3)
                
                c_category_loss = self.CELoss(c_category_logits_reshaped, batch_y[:,:,:1].view(-1).long())
                c_continuous_loss = self.MSELoss(c_continuous, batch_y[:,:,1:])
                c_loss = c_category_loss + c_continuous_loss
                
                # pixel loss for z
                infer_latent = self.model.get_infer_latent(encoder_output)
                decoded_enc = self.model.decode(infer_latent)
                z_loss = self.MSELoss(decoded_enc, batch)
                E_loss = c_loss + z_loss
                E_loss.backward()
                self.optimizer_E.step()
                

                                
                if i % 500 == 0:
                    print('Loss at epoch {} batch {}: D: {}, G {}, E {}'.format(e, i, D_loss, G_loss, E_loss))
                    np.save('/hdd_c/data/miniWorld/obs/rec.npy',decoded_fake.detach().cpu().numpy())
                    np.save('/hdd_c/data/miniWorld/obs/rec_enc.npy',decoded_enc.detach().cpu().numpy())
                    np.save('/hdd_c/data/miniWorld/obs/real.npy',batch.detach().cpu().numpy())

            #self.eval()
            save_model(self.model, self.model_path+'_epoch_{}_backup'.format(e))
            
    def eval(self, dataset=None, path='/hdd_c/data/miniWorld/obs/'):
        raise NotImplementedError
#         if dataset == None:
#             data = self.input_eval
#             y = self.y_eval
#         else:
#             raise NotImplementedError
            
#         self.model.eval()

#         num_batch = len(data)//self.batch_size
#         loss_list = []
#         with torch.no_grad():   
#             for i in range(num_batch):
#                 batch = data[i*self.batch_size:(i+1)*self.batch_size]
#                 batch = make_var(batch)
#                 y, mu, logsigma = self.model(batch)
#                 if i == 0:
#                     print(logsigma[0])
#                     np.save(path+'VAER_eval_reconstruction_batch_{}.npy'.format(i), y.detach().cpu())
#                     np.save(path+'VAER_eval_input_batch_{}.npy'.format(i), batch.detach().cpu())
#                 diff = y - batch
#                 loss = (diff * diff).mean() # L2 loss
#                 loss_list.append(loss)
#                 #if i % 50 == 0:
#                 #    print('Loss at epoch {} batch {}: {}'.format(e, i, loss))
#             print('Average L2 reconstruction loss on eval data: {}'.format(sum(loss_list)/len(loss_list)))





def get_args():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--lr', type=float, default=0.00005,
                        help='learning rate')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-dir', default='/hdd_c/data/miniWorld/log/',                   
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='/hdd_c/data/miniWorld/trained_models/',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    args = parser.parse_args()
    print(torch.cuda.is_available())
    print(args.no_cuda)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

args = get_args()



def read_data(path, max_num_eps = None):
    obs_list = []
    y_list = []
    filelist = os.listdir(path)
    num = 0
    for name in filelist:
        if name.endswith('.npz'):
            data = np.load(path+name)
            obs_list.append(data['obs'])
            ent = data['ent']
            y = process_ent(ent)
            y_list.append(y)
            num = num + 1
            if max_num_eps != None and num >= max_num_eps:
                break      
                
    all_obs = np.concatenate(obs_list, axis = 0)
    all_y = np.concatenate(y_list, axis = 0)
    return all_obs, all_y


def coor_trans(agent_pos, agent_dir, obj_pos):
    translated_obj_x = obj_pos[0] - agent_pos[0]
    translated_obj_z = obj_pos[2] - agent_pos[2]
    coor_rot = agent_dir
    new_obj_x = math.cos(coor_rot) * translated_obj_x - math.sin(coor_rot) * translated_obj_z
    new_obj_z = math.sin(coor_rot) * translated_obj_x + math.cos(coor_rot) * translated_obj_z
    new_vec_agent = [1, 0]
    new_vec_object = [new_obj_x, new_obj_z]
    new_vec_object_mag = math.sqrt(math.pow(new_obj_x, 2)+math.pow(new_obj_z, 2))

    new_obj_dir = math.acos((new_vec_agent[0]*new_vec_object[0]+new_vec_agent[1]*new_vec_object[1])/new_vec_object_mag)
    # determine left or right
    if new_obj_z >= 0:
        new_obj_dir = -1 * new_obj_dir
    return [new_obj_dir, new_vec_object_mag]

def process_ent(ent_seq):
    y = np.zeros([80,6,3])
    for i, ent_list in enumerate(ent_seq):
        idx = 0
        agent_dir = ent_list[-1, 4]
        agent_pos = ent_list[-1, 1:4]
        for j in range(6):
            obj_pos = ent_list[j, 1:4]
            posi_list = coor_trans(agent_pos, agent_dir, obj_pos) 
            if ent_list[j,-1]<0.5:
                if posi_list[0] >= -1*math.radians(30) and posi_list[0] <= math.radians(30):
                    y[i,idx,:] = np.asarray([ent_list[j, 0], posi_list[0], posi_list[1]])
                    idx = idx + 1
        if idx > 0:
            non_empty = y[i,:idx]
            sorted_non_empty = non_empty[non_empty[:,1].argsort()]
            y[i,:idx] = sorted_non_empty
    return y

def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    model = torch.load(path)
    return model



def main():

    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    print(device)
    
    
    model = SimGAN([128,128]).cuda()
    model_path = '/hdd_c/data/miniWorld/trained_models/SimGAN/dataset_5/SimGAN.pth'
    data_path = '/hdd_c/data/miniWorld/dataset_5/'
    all_obs, all_y = read_data(data_path, max_num_eps=1500)

    # normalize
    all_obs = all_obs/255.0
    all_obs = np.swapaxes(all_obs,1,3)
    y_max_dir = np.amax(all_y[:,:,1])
    y_min_dir = np.amin(all_y[:,:,1])
    y_max_dis = np.amax(all_y[:,:,2])
    y_min_dis = np.amin(all_y[:,:,2])
    print(y_max_dir)
    print(y_min_dir)
    print(y_max_dis)
    print(y_min_dis)
    all_y[:,:,1] = (all_y[:,:,1]-y_min_dir)/(y_max_dir-y_min_dir)
    all_y[:,:,2] = (all_y[:,:,2]-y_min_dis)/(y_max_dis-y_min_dis)
    
    # split
    split_point = int(all_obs.shape[0]*0.8)
    all_obs_train = all_obs[:split_point]
    all_obs_val = all_obs[split_point:]
    all_y_train = all_y[:split_point]
    all_y_val = all_y[split_point:]
    
    # train with shuffle
    indices = np.arange(all_obs_train.shape[0])
    np.random.shuffle(indices)
    all_obs_train = all_obs_train[indices]
    all_y_train = all_y_train[indices]
    # val with shuffle
    indices = np.arange(all_obs_val.shape[0])
    np.random.shuffle(indices)
    all_obs_val = all_obs_val[indices]
    all_y_val = all_y_val[indices]
    
    
    training_instance = trainSimGAN(device, model, lr=args.lr, eps=args.eps, input_train=all_obs_train, y_train=all_y_train, input_eval=all_obs_val, y_eval=all_y_val, model_path=model_path)
    training_instance.train()
    
    #training_instance.eval(all_obs[-10:])
    save_model(training_instance.model, model_path)

if __name__ == "__main__":
    main()