import os

starting_idx = 1205
n = 95
num_objs = 8
dataset_path = '/hdd_c/data/miniWorld/dataset_3/ '

for i in range(starting_idx, starting_idx+n):
    print(i)
    myCmd = 'python get_episode.py --path {} --idx {} --num_objs {}'.format(dataset_path, i, num_objs)
    os.system(myCmd)

# V1
# dataset_1: num_objs=1
# dataset_2: num_objs=8
# V2
# dataset_3: num_objs=8, obj size = 1.0
