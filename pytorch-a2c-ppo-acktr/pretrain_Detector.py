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
from model import Detector, resnet18
from storage import RolloutStorage
#from visualize import visdom_plot
import argparse
from utils import make_var
import math


class trainDetector():
    def __init__(self, device, model, lr, eps, input_train, y_train, input_eval, y_eval, model_path, batch_size = 32):
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)
        self.batch_size = batch_size
        self.input_train = input_train
        self.y_train = y_train
        self.input_eval = input_eval
        self.y_eval = y_eval
        self.BCELoss = nn.BCELoss()
        self.CELoss = nn.CrossEntropyLoss()
        self.MSELoss_reduction = nn.MSELoss()
        self.MSELoss = nn.MSELoss(reduction='none')
        self.model_path = model_path
        
        
    def train(self, dataset=None):
        if dataset == None:
            data = self.input_train
            y = self.y_train
        else:
            raise NotImplementedError
        self.model.train()
        #np.save('/hdd_c/data/miniWorld/obs/eval_input_batch_test_train.npy',data[:32])
        for e in range(15):
            num_batch = len(data)//self.batch_size
            #idx = 0
            for i in range(num_batch):
                batch = data[i*self.batch_size:(i+1)*self.batch_size]
                batch_y = y[i*self.batch_size:(i+1)*self.batch_size]
                batch = make_var(batch)
                batch_y = make_var(batch_y)
                
                self.optimizer.zero_grad()

                y_pred = self.model(batch)
                y_pred = y_pred.view(self.batch_size,6,5)
                
#                 if i == 0:
#                     print(batch_y)
#                     print(y_pred)
                    
                #batch_y = batch_y.view(self.batch_size,6,3)
                C_loss = self.CELoss(y_pred[:,:,:3].view(-1,3), batch_y[:,:,0].view(-1).long())
                R_loss = self.MSELoss(y_pred[:,:,3:], batch_y[:,:,1:])
                

                obj_mask = torch.ge(batch_y[:,:,0], 0.5)
                obj_mask = torch.unsqueeze(obj_mask, -1)
                #if i == 0:
                #print(obj_mask)
                #print(R_loss)
                #print(R_loss * obj_mask.float())
                #print(obj_mask.size())
                #print(R_loss.size())
                #print(obj_mask.view(32,6,1).size())
                R_loss = torch.mean(R_loss * obj_mask.float())
                if i == 0: print(R_loss)
                #print(y_pred[:,:,3:])
                #print(batch_y[:,:,1:])
                loss = C_loss + R_loss
                
                if i % 100 == 0:
                    print('Loss at epoch {} batch {}: {}, C_loss {}, R_loss {}'.format(e, i, loss, C_loss, R_loss))
                
                loss.backward()
                self.optimizer.step()
            self.eval()
            save_model(self.model, self.model_path+'_epoch_{}_backup'.format(e))
            
    def eval(self, dataset=None, path='/hdd_c/data/miniWorld/obs/'):
        if dataset == None:
            data = self.input_eval
            y = self.y_eval
        else:
            raise NotImplementedError
        self.model.eval()

        num_batch = len(data)//self.batch_size
        C_loss_list = []
        R_loss_list = []
        with torch.no_grad():   
            num_batch = len(data)//self.batch_size
            total_correct_num = 0.
            
            masked_list = [[], [], [], []]

            for i in range(num_batch):
                batch = data[i*self.batch_size:(i+1)*self.batch_size]
                batch_y = y[i*self.batch_size:(i+1)*self.batch_size]
                batch = make_var(batch)
                batch_y = make_var(batch_y)

                y_pred = self.model(batch)
                y_pred = y_pred.view(self.batch_size,6,5)
                
                if i == 0:
                    np.save('/hdd_c/data/miniWorld/obs/eval_input.npy', batch.detach().cpu().numpy())
                    np.save('/hdd_c/data/miniWorld/obs/eval_y.npy', batch_y.detach().cpu().numpy())
                    np.save('/hdd_c/data/miniWorld/obs/pred_y.npy', y_pred.detach().cpu().numpy())
                
                
                pred_idx = torch.max(y_pred[:,:,:3], dim=2)[1]
                
                correct_mask = torch.eq(pred_idx.view(-1), batch_y[:,:,0].view(-1).long())
                presence_mask = torch.ge(batch_y[:,:,0].view(-1), 0.5)
                obj_mask = correct_mask * presence_mask
                #print(obj_mask)
                #print(batch_y.view(-1,3)[obj_mask][:,1])
                #print(batch_y.view(-1,3)[:,1])
                masked_dir = batch_y.view(-1,3)[obj_mask][:,1]
                masked_dis = batch_y.view(-1,3)[obj_mask][:,2]
                masked_dir_pred = y_pred.view(-1,5)[obj_mask][:,3]
                masked_dis_pred = y_pred.view(-1,5)[obj_mask][:,4]
                
                masked_list[0].append(masked_dir)
                masked_list[1].append(masked_dis)
                masked_list[2].append(masked_dir_pred)
                masked_list[3].append(masked_dis_pred)

                
                #print(pred_idx)
                #batch_y = batch_y.view(self.batch_size,6,3)
                #print(batch_y[:,:,0])
                correct_num = torch.eq(pred_idx, batch_y[:,:,0].long()).sum()
                #print(correct_num)
                total_correct_num = total_correct_num + correct_num.detach().item()
                
                C_loss = self.CELoss(y_pred[:,:,:3].view(-1,3), batch_y[:,:,0].view(-1).long())
                R_loss = self.MSELoss(y_pred[:,:,3:], batch_y[:,:,1:])
                C_loss_list.append(C_loss)
                R_loss_list.append(R_loss)
            #self.eval()
            #save_model(self.model, self.model_path+'_epoch_{}_backup'.format(e))
            
            #dir_error = self.MSELoss_reduction(torch.cat()masked_dir_pred, masked_dir)
            #dis_error = self.MSELoss_reduction(masked_dis_pred, masked_dis)
            dir_error = self.MSELoss_reduction(torch.cat(masked_list[0]), torch.cat(masked_list[2]))
            dis_error = self.MSELoss_reduction(torch.cat(masked_list[1]), torch.cat(masked_list[3]))
            print(total_correct_num)
            print('acc: {}'.format(total_correct_num/float(num_batch*32*6)))
            print('Average loss on eval data: C loss {}, dir_error {}, dis_error {}'.format(sum(C_loss_list)/len(C_loss_list),dir_error,dis_error))





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
            #print(math.radians(60))
            if ent_list[j,-1]<0.5:
                if posi_list[0] >= -1*math.radians(30) and posi_list[0] <= math.radians(30):
                    y[i,idx,:] = np.asarray([ent_list[j, 0], posi_list[0], posi_list[1]])
                    #y[i,idx*3:(idx+1)*3] = np.asarray([ent_list[j, 0], posi_list[0], posi_list[1]])
                    idx = idx + 1
                
                #print(posi_list)
        #print(y[i])        
        # sort from right to left
        #print(y)
        if idx > 0:
            non_empty = y[i,:idx]
            sorted_non_empty = non_empty[non_empty[:,1].argsort()]
            y[i,:idx] = sorted_non_empty
    #y = np.reshape(y,(-1,18))
    return y

def save_model(model, path):
    torch.save(model, path)

def load_model(path):
    model = torch.load(path)
    return model



def main():

    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    print(device)
    #model = VAE([128,128], lr=args.lr, eps=args.eps)
    
    model = Detector()
    
    model = resnet18(pretrained=False, progress=False)
    
    
    #model_path = '/hdd_c/data/miniWorld/trained_models/Detector/dataset_5/Detector.pth'
    model_path = '/hdd_c/data/miniWorld/trained_models/Detector/dataset_5/Detector_resnet18.pth'
    
    data_path = '/hdd_c/data/miniWorld/dataset_5/'
    all_obs, all_y = read_data(data_path, max_num_eps=400)

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
    #print(all_y[:10,:,:])
    #raise Error
    
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
    

    
    print('Available number of obs: {}'.format(len(all_obs)))
    print(all_obs_train.shape)
    
    
    #data_train = all_obs[:200000]
    #data_eval = all_obs[200000:240000]
    #image = np.zeros([32,3,128,128])
    #image = make_var(image)
    #z = model.encode(image)
    #r = model.decode(z)
    #dummy_data = np.ones([6400,3,128,128])
    #print(data_eval.shape)
    training_instance = trainDetector(device, model, lr=args.lr, eps=args.eps, input_train=all_obs_train, y_train=all_y_train, input_eval=all_obs_val, y_eval=all_y_val, model_path=model_path)
    training_instance.train()
    
    #training_instance.eval(all_obs[-10:])
    #save_model(training_instance.model, model_path)

if __name__ == "__main__":
    main()