

import sys
import argparse
import pyglet
import math
from pyglet.window import key
from pyglet import clock
import numpy as np
import gym
import gym_miniworld
import matplotlib.pyplot as plt
from IPython import display
from IPython.display import clear_output
import time
import random

#env = gym.make('MiniWorld-Hallway-v0')
env = gym.make('MiniWorld-Navigation-v0', num_objs=20)


#MiniWorld-Hallway-v0

# if args.no_time_limit:
#     env.max_episode_steps = math.inf
# if args.domain_rand:
#     env.domain_rand = True

env.reset()

left_turn_tendency = random.randint(0,1)
print(left_turn_tendency)



def rand_act_1(env, act_list):
    r = random.randint(0,99)
    if r < 25:
        act_list.append(env.actions.turn_left if left_turn_tendency == 1 else env.actions.turn_right)
    elif r < 35:
        act_list.append(env.actions.turn_right if left_turn_tendency == 1 else env.actions.turn_left)
    else:
        act_list.append(env.actions.move_forward)
    return act_list
        

def step(action):
    

    print('step {}/{}: {}'.format(env.step_count+1, env.max_episode_steps, env.actions(action).name))

    obs, reward, done, info = env.step(action)

    if reward > 0:
        print('reward={:.2f}'.format(reward))
    if done:
        print('done!')
        env.reset()

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
    
    
    #print(obs.shape)
    obs_list.append(obs)
    ren = env.render(mode='rgb_array')
    ren_list.append(ren)
    top = env.render_top_view(view_size = [2000, 2000])
    top_list.append(top)
    ent_info_list.append(ent_info)
    
act_list = []
action_space = [env.actions.move_forward, env.actions.move_back, env.actions.turn_left, env.actions.turn_right]

obs_list = []
ren_list = []
top_list = []
ent_info_list = []
for _ in range(40):
    #print(env.params['forward_step'])
    if len(act_list) == 0:
        act_list = rand_act_1(env, act_list)
    #act = random.choice(action_space)
    step(act_list[0])
    del act_list[0]
env.reset()
obs_list = np.asarray(obs_list)
top_list = np.asarray(top_list)
ren_list = np.asarray(ren_list)
ent_info_list = np.asarray(ent_info_list)

np.savez_compressed('/hdd_c/data/miniWorld/obs/test.npz', obs = obs_list, top = top_list, ren = ren_list, ent = ent_info_list)


env.close()

