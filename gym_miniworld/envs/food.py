import numpy as np
import math
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Ball, Key, Medkit, Cone, Building, Duckie
from gym import spaces

class Food(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    at the end of a hallway
    """

    def __init__(self, size=15,mode='normal', **kwargs):
        assert size >= 4
        self.size = size
        self.ent_list = []
        self.mode = mode # data, normal, real
        self.positive_obj_num = 3
        self.positive_ent_list = []
        self.remaining_obj = 6
        self.accumulated_reward = 0

        super().__init__(
            max_episode_steps=80, obs_width = 128, obs_height = 128, window_width = 1280, window_height = 1280,
            **kwargs
        )
        
        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward+1)

    def _gen_world(self):
        # reset accumulated_reward
        self.accumulated_reward = 0
        
        # Create a long rectangular room
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex='brick_wall',
            floor_tex='asphalt',
            no_ceiling=True,
        )
        self.ent_list = []
#     'red'   : np.array([1.0, 0.0, 0.0]),
#     'green' : np.array([0.0, 1.0, 0.0]),
#     'blue'  : np.array([0.0, 0.0, 1.0]),
#     'purple': np.array([0.44, 0.15, 0.76]),
#     'yellow': np.array([1.00, 1.00, 0.00]),
#     'grey'  : np.array([0.39, 0.39, 0.39])
        
        #if self.mode == 'data':
        self.b_1 = self.place_entity( Ball(color='red', size = 0.5) )
        self.b_2 = self.place_entity( Ball(color='green', size = 0.5) )
        self.b_3 = self.place_entity( Ball(color='blue', size = 0.5) )
        self.c_1 = self.place_entity( Box(color='red', size = 0.5) )
        self.c_2 = self.place_entity( Box(color='green', size = 0.5) )
        self.c_3 = self.place_entity( Box(color='blue', size = 0.5) )

        self.ent_list.append(self.b_1)
        self.ent_list.append(self.b_2)
        self.ent_list.append(self.b_3)
        self.ent_list.append(self.c_1)
        self.ent_list.append(self.c_2)
        self.ent_list.append(self.c_3)
                    
#         # Generate special objects
#         obj_types = [Ball, Key, Medkit, Cone, Building, Duckie]
#         #obj_types = [Box,]

#         for obj_type in obj_types:
#             #obj_type = self.rand.choice(obj_types)
#             color = self.rand.color()

#             if obj_type == Box:
#                 ent = self.place_entity(Box(color=color, size=[0.2,1,1]))
#                 #print(ent.pos)
#                 #print(ent.dir)
#             if obj_type == Ball:
#                 ent = self.place_entity(Ball(color='blue'))
#             if obj_type == Key:
#                 ent = self.place_entity(Key(color='yellow'))
#             if obj_type == Building:
#                 ent = self.place_entity(Building(color='yellow'))
#             if obj_type == Cone:
#                 ent = self.place_entity(Cone(color='yellow'))
#             if obj_type == Medkit:
#                 ent = self.place_entity(Medkit(color='yellow'))
#             if obj_type == Duckie:
#                 ent = self.place_entity(Duckie(color='yellow'))
#             #self.ent_list.append(ent)
            
            
        # Place the agent a random distance away from the goal
        ent = self.place_agent()
        self.ent_list.append(ent)
        
        
    def step(self, action):
        obs, reward, done, info = super().step(action)
        #print(self.ent_list[-1].pos[0])
        #print(self.ent_list[-1].pos[2])
        if self.mode == 'data':
            info['ent_list'] = self.ent_list
        
        #print(done)
        #print(len())
        for ent in self.ent_list[:-1]:
            if ent.is_removed == False and self.near(ent):
                if ent.__name__ == 'Ball':
                    reward += 1
                else:
                    reward -= 1
                #print(reward)
                self.entities.remove(ent)
                self.remaining_obj -= 1
                ent.is_removed = True
        self.accumulated_reward += reward
        info['accumulated_reward'] = self.accumulated_reward
        #reward += 1
        if self.remaining_obj == 0:
            done = True
            #done = True
        #if self.near(self.box2):
        #    reward += -1
        #    done = True

        return obs, reward, done, info
