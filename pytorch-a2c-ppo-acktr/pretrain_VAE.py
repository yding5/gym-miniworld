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
from model import VAE
from storage import RolloutStorage
#from visualize import visdom_plot
import argparse
from utils import make_var


class trainVAE():
    def __init__(self, model, lr, eps, batch_size = 32):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)
        self.batch_size = batch_size
        
    def train(self, data):
        for e in range(2):
            num_batch = len(data)//self.batch_size
            #idx = 0
            for i in range(num_batch):
                batch = data[i*self.batch_size:(i+1)*self.batch_size]
                batch = make_var(batch)
                self.optimizer.zero_grad()

                z = self.model.encode(batch)
                y = self.model.decode(z)
                diff = y - batch

                loss = (diff * diff).mean() # L2 loss
                
                if i % 50 == 0:
                    print('Loss at epoch {} batch {}: {}'.format(e, i, loss))
                
                loss.backward()
                self.optimizer.step()
                


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

def main():

    device = torch.device("cuda:0" if args.cuda else "cpu")
    print(device)
    model = VAE([128,128], lr=args.lr, eps=args.eps)
    image = np.zeros([32,3,128,128])
    image = make_var(image)
    z = model.encode(image)
    r = model.decode(z)
    
    dummy_data = np.ones([6400,3,128,128])
    training_instance = trainVAE(model, lr=args.lr, eps=args.eps)
    training_instance.train(dummy_data)



if __name__ == "__main__":
    main()