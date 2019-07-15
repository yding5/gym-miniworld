import numpy as np
import math
from gym import spaces
from ..miniworld import MiniWorldEnv, Room
from ..entity import Box, Ball, Key, Medkit, Cone, Building, Duckie

#         obs_width=80,
#         obs_height=60,
#         window_width=800,
#         window_height=600,

class Navigation(MiniWorldEnv):
    """
    Room with multiple objects. The agent collects +1 reward for picking up
    each object. Objects disappear when picked up.
    """

    def __init__(self, size=20, num_objs=5, **kwargs):
        assert size >= 2
        self.size = size
        self.num_objs = num_objs
        
        # Entity in the env
        self.ent_list = []

        super().__init__(
            max_episode_steps=400, obs_width = 128, obs_height = 128, window_width = 1280, window_height = 1280,
            **kwargs
        )

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.move_back+1)
        


    def _gen_world(self):
        
        self.ent_list = []
        
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex='brick_wall',
            floor_tex='asphalt',
            no_ceiling=True,
        )
        
        
        # Generate walls
        for i in range(3):
            #obj_type = self.rand.choice(obj_types)
            color = self.rand.color()
            ent = self.place_entity(Box(color=color, size=[0.2,2,6]))
            self.ent_list.append(ent)
            
            
        # Generate special objects
        obj_types = [Ball, Key, Medkit, Cone, Building, Duckie]

        for obj in range(self.num_objs):
            obj_type = self.rand.choice(obj_types)
            color = self.rand.color()

            if obj_type == Box:
                ent = self.place_entity(Box(color=color, size=[0.2,2,8]))
                #print(ent.pos)
                #print(ent.dir)
            if obj_type == Ball:
                ent = self.place_entity(Ball(color='blue', size=0.9))
            if obj_type == Key:
                ent = self.place_entity(Key(color='yellow'))
            if obj_type == Building:
                ent = self.place_entity(Building(color='yellow'))
            if obj_type == Cone:
                ent = self.place_entity(Cone(color='yellow'))
            if obj_type == Medkit:
                ent = self.place_entity(Medkit(color='yellow'))
            if obj_type == Duckie:
                ent = self.place_entity(Duckie(color='yellow'))
            self.ent_list.append(ent)
            
            
        ent = self.place_agent()
        self.ent_list.append(ent)
        
        #print('agent pos and dir:')
        #print(ent.pos)
        #print(ent.dir)

        self.num_picked_up = 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        info['ent_list'] = self.ent_list

        if self.agent.carrying:
            self.entities.remove(self.agent.carrying)
            self.agent.carrying = None
            self.num_picked_up += 1
            reward = 1

            if self.num_picked_up == self.num_objs:
                done = True

        return obs, reward, done, info
