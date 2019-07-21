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
from model import VAE, VAEU, RNN
from storage import RolloutStorage
#from visualize import visdom_plot
import argparse
from utils import make_var


class trainRNN():
    def __init__(self, device, model, lr, eps, x_train, y_train, x_val, y_val, batch_size = 32, num_type = 6):
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val 
        self.y_val = y_val 
        self.BCELoss = nn.BCELoss()
        self.CEloss = nn.CrossEntropyLoss()
        self.CEloss_none_reduction = nn.CrossEntropyLoss(reduction='none')
        self.L1Loss = nn.L1Loss()
        self.MSELoss = torch.nn.MSELoss()
        self.num_type = num_type
        
        self.pred_loss_weight = 1.0
        self.discounted_pred_loss_weight = 1.0
        self.conf_loss_weight = 1.0
        
    def train(self, x_train=None, y_train=None):
        if x_train is None:
            x_train = self.x_train
            y_train = self.y_train
        
        num_sample = x_train.shape[1]
        
        for e in range(100):
            self.model.train()
            x_train, y_train = unison_shuffled_copies_dim_1(x_train, y_train)
            num_batch = num_sample//self.batch_size
            #idx = 0
            for i in range(num_batch):
                #print(i)
                x = x_train[:,i*self.batch_size:(i+1)*self.batch_size]
                x = make_var(x)
                y = y_train[:,i*self.batch_size:(i+1)*self.batch_size]
                y = make_var(y).long()

                self.optimizer.zero_grad()

                obj, conf = self.model(x)

                
                _, pred_y = torch.max(obj, dim=2)
                correct = torch.eq(pred_y.detach(), y).float()
                batch_acc = torch.mean(correct, dim=1).view(-1)
                
                batch_conf = torch.mean(conf, dim=1).view(-1)
                

                obj = obj.view(-1,self.num_type)
                y = y.view(-1)
                conf = conf.view(-1)
                
                pred_loss = self.CEloss(obj, y)
                pred_loss_none_reduction = self.CEloss_none_reduction(obj, y)
                pred_loss_none_reduction = pred_loss_none_reduction.view(-1)

                discounted_pred_loss = torch.mean(pred_loss_none_reduction * conf)

                
                conf_loss = self.MSELoss(conf, torch.ones_like(conf))

                total_loss = self.pred_loss_weight * pred_loss + self.discounted_pred_loss_weight * discounted_pred_loss + self.conf_loss_weight * conf_loss 

                total_loss.backward()
                self.optimizer.step()

                
                if i % 50 == 0:
                    print('Epoch {} batch {}: Total loss {}, Dis_pred_loss {}, conf_loss {}'.format(e, i, total_loss, discounted_pred_loss, conf_loss))
                    #print(batch_acc)
                    #print(batch_conf)

            print('Epoch {}: eval on training set:'.format(e))
            self.eval(x_val=x_train, y_val=y_train)
            print('Epoch {}: eval on validation set:'.format(e))
            self.eval()
            
    def eval(self, x_val=None, y_val=None, path='/hdd_c/data/miniWorld/obs/'):

        if x_val is None:
            x_val = self.x_val
            y_val = self.y_val
        self.model.eval()
        
        acc_list = []
        conf_list = []
        loss_list = []
        
        num_sample = x_val.shape[1]
        with torch.no_grad():
            num_batch = num_sample//self.batch_size
            #idx = 0
            for i in range(num_batch):
                #print(i)
                x = x_val[:,i*self.batch_size:(i+1)*self.batch_size]
                x = make_var(x)
                y = y_val[:,i*self.batch_size:(i+1)*self.batch_size]
                y = make_var(y).long()

                obj, conf = self.model(x)

                _, pred_y = torch.max(obj.detach(), dim=2)
                correct = torch.eq(pred_y, y).float()
                batch_acc = torch.mean(correct, dim=1).view(-1).cpu()
                
                batch_conf = torch.mean(conf.detach(), dim=1).view(-1).cpu()

                obj = obj.view(-1,self.num_type)
                y = y.view(-1)
                conf = conf.view(-1)

                pred_loss = self.CEloss(obj, y)
                pred_loss_none_reduction = self.CEloss_none_reduction(obj, y)
                #print(pred_loss_none_reduction.size())
                pred_loss_none_reduction = pred_loss_none_reduction.view(-1)
                #print(pred_loss_none_reduction.size())
                discounted_pred_loss = torch.mean(pred_loss_none_reduction * conf)
                #conf_aware_pred_loss = torch.mean(pred_loss_none_reduction * conf)

                conf_loss = self.MSELoss(conf, torch.ones_like(conf))
                #torch.ones_like(conf)
                total_loss = self.pred_loss_weight * pred_loss + self.discounted_pred_loss_weight * discounted_pred_loss + self.conf_loss_weight * conf_loss 
                
                acc_list.append(np.asarray(batch_acc))
                conf_list.append(np.asarray(batch_conf))
                loss_list.append(np.asarray([total_loss.cpu(), pred_loss.cpu(), discounted_pred_loss.cpu(), conf_loss.cpu()]))
            eval_acc = np.mean(np.asarray(acc_list), axis=0)
            eval_conf = np.mean(np.asarray(conf_list), axis=0)
            eval_loss = np.mean(np.asarray(loss_list), axis=0)
            print('Eval results: loss {}, acc[0] {}, acc[-1] {}, conf[0] {}, conf[-1] {}'.format(eval_loss, eval_acc[0],eval_acc[-1],eval_conf[0],eval_conf[-1]))
#             print(eval_acc[0])
#             print(eval_acc[-1])
#             print(eval_conf[0])
#             print(eval_conf[-1])
#             print(eval_loss)



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
            data = np.load(path+name)
            obs_list.append(data['obs'])
    all_obs = np.concatenate(obs_list, axis = 0)
    return all_obs

def read_z_data(path):
    z_list = []
    obj_type_list = []
    filelist = os.listdir(path)
    for name in filelist:
        if name.endswith('.npz'):
            data = np.load(path+name)
            z_list.append(data['z'])
            obj_type_list.append(data['obj_type'])
    all_z = np.stack(z_list, axis = 1)# (seq_length, number_seq, dim)
    all_obj_type = np.stack(obj_type_list, axis = 1) # (seq_length, number_seq)
    return all_z, all_obj_type

def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    model = torch.load(path)
    return model


def unison_shuffled_copies_dim_1(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a[0]))
    return a[:,p], b[:,p]


def main():

    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    print(device)
    #model = VAE([128,128], lr=args.lr, eps=args.eps)
    model = RNN(200, 128)
    model_path = '/hdd_c/data/miniWorld/trained_models/RNN/RNN1.pth'
    data_path = '/hdd_c/data/miniWorld/z_dataset_1/'

    all_z, all_obj_type = read_z_data(data_path)
    all_obj_type = all_obj_type - 2 * np.ones_like(all_obj_type)
    print(np.amax(all_obj_type))
    print(np.amin(all_obj_type))
    print('Available number of seq: {}'.format(all_z.shape[1]))
    print(all_z.shape)
    x_train = all_z[:,:2500]
    y_train = all_obj_type[:,:2500]
    print(y_train[0,0])
    x_val = all_z[:,2500:]
    y_val = all_obj_type[:,2500:]

    
    print(x_train.shape)
    training_instance = trainRNN(device, model, lr=args.lr, eps=args.eps, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)
    training_instance.train()
    
    save_model(training_instance.model, model_path)


if __name__ == "__main__":
    main()