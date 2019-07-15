import sys
import argparse
import pyglet
import math
#from pyglet.window import key
#from pyglet import clock
import numpy as np
import gym
import gym_miniworld
#import matplotlib.pyplot as plt
#from IPython import display
#from IPython.display import clear_output
import time

import random



# if args.domain_rand:
#     env.domain_rand = True

def rand_act_1(env, act_list, left_turn_tendency):
    r = random.randint(0,99)
    if r < 25:
        act_list.append(env.actions.turn_left if left_turn_tendency == 1 else env.actions.turn_right)
    elif r < 35:
        act_list.append(env.actions.turn_right if left_turn_tendency == 1 else env.actions.turn_left)
    else:
        act_list.append(env.actions.move_forward)
    return act_list


def step(env, action):
    print('step {}/{}: {}'.format(env.step_count+1, env.max_episode_steps, env.actions(action).name))

    obs, reward, done, info = env.step(action)
    
    ren = env.render(mode='rgb_array')
    top = env.render_top_view(view_size = [1280, 1280])

    ent_info = []
    for ent in info['ent_list']:
        #print(ent)
        if ent.__name__ == 'Agent':
            ent_name = 0
        elif ent.__name__ == 'Box':
            ent_name = 1
        elif ent.__name__ == 'Ball':
            ent_name = 2
        elif ent.__name__ == 'Key':
            ent_name = 3
        elif ent.__name__ == 'Medkit':
            ent_name = 4
        elif ent.__name__ == 'Cone':
            ent_name = 5
        elif ent.__name__ == 'Building':
            ent_name = 6
        elif ent.__name__ == 'Duckie':
            ent_name = 7
        else:
            raise NotImplementedError('unknown object type')
        if ent_name == 1:
            ent_info.append([ent.pos[0], ent.pos[1], ent.pos[2], ent.dir, ent.size[0], ent.size[1], ent.size[2]])
        else:
            ent_info.append([ent.pos[0], ent.pos[1], ent.pos[2], ent.dir, ent.radius, ent.height, ent.radius])   
    ent_info = np.asarray(ent_info)

    if reward > 0:
        print('reward={:.2f}'.format(reward))
    if done:
        print('done!')
        env.reset()
    return obs, ren, top, ent_info

def get_one_episode(env, path, idx):

    env.reset()

    # Random Action
    left_turn_tendency = random.randint(0,1)

    print('left_turn_tendency: {}'.format(left_turn_tendency))

    obs_seq = []
    ren_seq = []
    top_seq = []
    ent_info_seq = []

    act_list = []

    for _ in range(80):
        if len(act_list) == 0:
            act_list = rand_act_1(env, act_list, left_turn_tendency)
        obs, ren, top, ent_info = step(env, act_list[0])
        
        obs_seq.append(obs)
        ren_seq.append(ren)
        top_seq.append(top)
        ent_info_seq.append(ent_info)
        
        del act_list[0]

    obs_seq = np.asarray(obs_seq)
    top_seq = np.asarray(top_seq)
    ren_seq = np.asarray(ren_seq)
    ent_info_seq = np.asarray(ent_info_seq)
    
    # rendering is not saved to save space
    np.savez_compressed(path+'eps_{}.npz'.format(idx), obs = obs_seq, top = top_seq, ent = ent_info_seq) 

    #return obs_seq, top_seq, ren_seq, ent_info_seq



# if args.no_time_limit:
#     env.max_episode_steps = math.inf

path = '/hdd_c/data/miniWorld/dataset_1/'
print('saving to {}'.format(path))

# obs_eps = []
# ren_eps = []
# top_eps = []
# ent_info_eps = []



for idx_episode in range(1000):
    env = gym.make('MiniWorld-Navigation-v0', num_objs=1)
    print('start episode {}'.format(idx_episode))
    get_one_episode(env, path, idx_episode)
    env.close()
    #obs_seq, top_seq, ren_seq, ent_info_seq = get_one_episode(env)
    #obs_eps.append(obs_seq)
    #top_eps.append(top_seq)
    #ren_eps.append(ren_seq)
    #ent_info_eps.append(ent_info_seq)



#obs_eps = np.asarray(obs_eps)
#top_eps = np.asarray(top_eps)
#ren_eps = np.asarray(ren_eps)
#ent_info_eps = np.asarray(ent_info_eps)

# rendering is not saved to save space
#np.savez_compressed('/hdd_c/data/miniWorld/dataset/data_{}.npz'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), obs = obs_eps, top = top_eps, ent = ent_info_eps) 
