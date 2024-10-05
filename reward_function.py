import numpy as np
import math
import argparse


# # ------------------------ Reward Function  ---------------------------

def get_reward(l2_dis_t, l2_dis_t_1, step):
    k = 1.0
    alpha = 0.8

    if l2_dis_t - l2_dis_t_1 > 0:
        reward = math.exp(-(k - alpha / (step + 1)) * l2_dis_t_1)
    else:
        reward = 0

    return reward



