import os

starting_idx = 5800
n = 200
num_objs = 0
dataset_path = '/hdd_c/data/miniWorld/dataset_5/ '
print('saving to {}'.format(dataset_path))

for i in range(starting_idx, starting_idx+n):
    print(i)
    myCmd = 'python get_episode_food.py --path {} --idx {} --num_objs {}'.format(dataset_path, i, num_objs)
    os.system(myCmd)

# V1
# dataset_1: num_objs=1
# dataset_2: num_objs=8
# V2
# dataset_3: num_objs=8, obj size = 1.0
# V3
# dataset_4: env: Food, balls and cubes size=0.5, red green blue 6 objects
# dataset_5: change the camera height/pitch to xxx, others are the same with dataset_4
