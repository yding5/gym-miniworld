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
from model import VAE, VAEU
from storage import RolloutStorage
#from visualize import visdom_plot
import argparse
from utils import make_var


class trainVAE():
    def __init__(self, device, model, lr, eps, data_train, data_eval, model_path, batch_size = 32):
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)
        self.batch_size = batch_size
        self.data_train = data_train
        self.data_eval = data_eval
        self.BCELoss = nn.BCELoss()
        self.model_path = model_path
        
    def train(self, data=None):
        if data == None:
            data = self.data_train
        self.model.train()
        #np.save('/hdd_c/data/miniWorld/obs/eval_input_batch_test_train.npy',data[:32])
        for e in range(15):
            num_batch = len(data)//self.batch_size
            #idx = 0
            for i in range(num_batch):
                batch = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = make_var(batch)
                self.optimizer.zero_grad()

                z = self.model.encode(batch)
                y = self.model.decode(z)
                #print(y.size())
                loss = self.BCELoss(y, batch)
                
                if i % 100 == 0:
                    print('Loss at epoch {} batch {}: {}'.format(e, i, loss))
                
                loss.backward()
                self.optimizer.step()
            self.eval()
            save_model(self.model, self.model_path+'_epoch_{}_backup'.format(e))
            
    def eval(self, data=None, path='/hdd_c/data/miniWorld/obs/'):
        if data == None:
            data = self.data_eval
        self.model.eval()

        num_batch = len(data)//self.batch_size
        loss_list = []
        with torch.no_grad():   
            for i in range(num_batch):
                batch = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = make_var(batch)
                z = self.model.encode(batch)
                y = self.model.decode(z)
                if i == 0:
                    np.save(path+'VAEU_eval_reconstruction_batch_{}.npy'.format(i), y.detach().cpu())
                    np.save(path+'VAEU_eval_input_batch_{}.npy'.format(i), batch.detach().cpu())
                diff = y - batch
                loss = (diff * diff).mean() # L2 loss
                loss_list.append(loss)
                #if i % 50 == 0:
                #    print('Loss at epoch {} batch {}: {}'.format(e, i, loss))
            print('Average L2 reconstruction loss on eval data: {}'.format(sum(loss_list)/len(loss_list)))





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
    #parser.add_argument('--save-dir', default='./trained_models/',
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
    filelist = os.listdir(path)
    num = 0
    for name in filelist:
        if name.endswith('.npz'):
            #print('reading {}'.format(name))
            data = np.load(path+name)
            obs_list.append(data['obs'])
            num = num + 1
            if max_num_eps != None and num >= max_num_eps:
                break
    all_obs = np.concatenate(obs_list, axis = 0)
    return all_obs

def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    model = torch.load(path)
    return model



def main():

    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    print(device)
    #model = VAE([128,128], lr=args.lr, eps=args.eps)
    model = VAEU([128,128])
    model_path = '/hdd_c/data/miniWorld/trained_models/VAE/dataset_4/VAEU.pth'
    data_path = '/hdd_c/data/miniWorld/dataset_4/'
    all_obs = read_data(data_path, max_num_eps=3000)
    np.random.shuffle(all_obs)
    all_obs = np.swapaxes(all_obs,1,3)
    all_obs = all_obs/255.0
    print('Available number of obs: {}'.format(len(all_obs)))
    print(all_obs.shape)
    data_train = all_obs[:96000]
    data_eval = all_obs[96000:128000]
    #image = np.zeros([32,3,128,128])
    #image = make_var(image)
    #z = model.encode(image)
    #r = model.decode(z)
    #dummy_data = np.ones([6400,3,128,128])
    print(data_eval.shape)
    training_instance = trainVAE(device, model, lr=args.lr, eps=args.eps, data_train=data_train, data_eval=data_eval, model_path=model_path)
    training_instance.train()
    
    #training_instance.eval(all_obs[-10:])
    save_model(training_instance.model, model_path)

if __name__ == "__main__":
    main()