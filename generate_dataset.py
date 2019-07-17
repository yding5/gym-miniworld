import os

starting_idx = 700
n = 1000

for i in range(starting_idx, starting_idx+n):
    print(i)
    myCmd = 'python get_episode.py --path /hdd_c/data/miniWorld/dataset_1/ --idx {}'.format(i)
    os.system(myCmd)