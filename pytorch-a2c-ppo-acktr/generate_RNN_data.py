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
import random




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


def read_data(path):
    obs_list = []
    filelist = os.listdir(path)
    for name in filelist:
        if name.endswith('.npz'):
            #print('reading {}'.format(name))
            data = np.load(path+name)
            obs_list.append(data['obs'])
    all_obs = np.concatenate(obs_list, axis = 0)
    return all_obs

def read_eps_data(path):
    obs_list = []
    idx_list = []
    type_list = []
    filelist = os.listdir(path)
    for name in filelist:
        splited = name.split('.')
        idx = splited[0][4:]
        #print(idx)
        if name.endswith('.npz'):
            #print('reading {}'.format(name))
            data = np.load(path+name)
            obs = data['obs']
            obj_type = data['ent'][:,3,0]
            obs = np.swapaxes(obs,1,3)/255.0
            obs_list.append(obs)
            idx_list.append(idx)
            type_list.append(obj_type)
    return obs_list, idx_list, type_list

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
    model_path = '/hdd_c/data/miniWorld/trained_models/VAE/VAEU.pth'
    data_path = '/hdd_c/data/miniWorld/dataset_1/'
    z_data_path = '/hdd_c/data/miniWorld/z_dataset_1/'
    
    
    eps_list, idx_list, type_list = read_eps_data(data_path)
    #eps_list = random.shuffle(eps_list)
    
    
    #all_obs = read_data(data_path)
    #np.random.shuffle(all_obs)
    #all_obs = np.swapaxes(all_obs,1,3)
    #all_obs = all_obs/255.0
    print('Available number of episode: {}'.format(len(eps_list)))
    
    #data_train = all_obs[:96000]
    #data_eval = all_obs[96000:128000]
    
    model = load_model(model_path)
    model.eval()
    for eps, idx, obj_type in zip(eps_list, idx_list, type_list):
        obs = make_var(eps)
        z = model.encode(obs)
        z_numpy = z.detach().cpu()
        #print(z_numpy.shape)
        #print(obj_type.shape)
        np.savez_compressed(z_data_path+'eps_{}.npz'.format(idx), z = z_numpy, obj_type = obj_type)
        #np.save(z_data_path+'obj_type_{}.npy'.format(idx), obj_type)
    
    #training_instance = trainVAE(device, model, lr=args.lr, eps=args.eps, data_train=data_train, data_eval=data_eval)
    #training_instance.train()
    
    #save_model(training_instance.model, model_path)
    #training_instance.eval(all_obs[-10:])


if __name__ == "__main__":
    main()