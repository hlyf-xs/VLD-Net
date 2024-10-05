import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
import argparse


from Environment_Scale_Size import Environment
from VB_Loc_Agent import Agent

from reward_function import get_reward


def get_args_parser():
    parser = argparse.ArgumentParser('VB localization', add_help=False)
    parser.add_argument('--MAX_EPISODES', default=110000, type=int,
                        help='Max episodes')
    parser.add_argument('--MAX_EP_STEPS', default=17, type=int)
    parser.add_argument('--per_img_explore_num', default=20, type=int)

    parser.add_argument('--out_graph', default=True, type=bool)

    parser.add_argument('--train_img_dir', default='data/images', type=str)
    parser.add_argument('--display_dir', default='display/', type=str)


    return argparse

def main(args):
    # -----------------   hyper parameters -------------------------
    Color_List = ['r', 'g', 'b', 'y', 'w']

    MAX_EPISODES = args.MAX_EPISODES
    MAX_EP_STEPS = args.MAX_EP_STEPS

    train_img_dir = args.train_img_dir
    train_img_list = os.listdir(train_img_dir)

    Train_img_num = len(train_img_list)

    per_img_explore_num = args.per_img_explore_num

    out_graph = args.per_img_explore_num
    display_dir = args.display_dir

    #  ------------------   Agent   --------------------------------
    env = Environment()
    Agent = Agent(out_graph=out_graph)

    train_step = 0


    for epoch_idx in range(MAX_EPISODES):
        print('\n Episode : --------------------- [', epoch_idx, '] --------------------')

        for img_idx in tqdm(range(Train_img_num)):

            print("img_idx:", img_idx)

            var_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0]

            for peer_img_loop_num in range(per_img_explore_num):
                # print('\nimg id : [{0}] --- explore_loop : [ {1} ]'.format(train_img_list[img_idx], peer_img_loop_num+1))
                print('\033[1;31;31m- img id : [{0}] --- explore_loop : [ {1} ]\033[0m'.format(train_img_list[img_idx],
                                                                                               peer_img_loop_num+1))
                # print("img_idx:", img_idx)
                plt_fig = False

                # start point, start state!
                p, s = env.reset(img_idx=img_idx)
                p_gt = env.get_label_i(vb_idx=0)

                # need to get patch
                if (train_step > 0) and train_step % 100 == 0:
                    plt_fig = True

                if plt_fig:
                    _, img = env.data_process.load_img(env.img_idx)
                    plt.figure(train_step)
                    plt.imshow(img)

                step_count = 0
                reward_list = []

                print('var_list : [ {0} ]'.format(var_list))

                # j : step index   { img i } :
                for j in range(MAX_EP_STEPS):

                    s_input = env.make_s_input(s)
                    # print("s_input:", s_input.shape)

                    # j-th point
                    p_gt_t = env.get_label_i(vb_idx=j)
                    l2_dis_t = np.linalg.norm(p - p_gt_t)

                    #  --------------------------  Agent choose Action
                    if plt_fig:
                        a = Agent.choose_action(s_input)
                        if math.isnan(a[0][0]) or math.isnan(a[0][1]):
                            break
                    else:
                        a = Agent.choose_action(s_input)
                        if math.isnan(a[0][0]) or math.isnan(a[0][1]):
                            break
                        a_norm = np.random.normal(loc=a, scale=var_list[j])
                        a = np.clip(a_norm, -1, 1)

                    #  ------------------   Agent execute Action
                    # modify this!
                    p_, s_ = env.step(a, p)
                    step_count += 1

                    s_input_ = env.make_s_input(s_)

                    p, s = p_, s_
                    p_gt = env.get_label_i(vb_idx=j+1)

                    # j+1-th point
                    l2_dis_t_1 = np.linalg.norm(p - p_gt)
                    var_list[j] = l2_dis_t_1 / 60.0

                    r_new = get_reward(l2_dis_t, l2_dis_t_1, step=j)

                    reward_list.append(round(r_new, 5))

                    if plt_fig:
                        plt.plot(p_[0], p_[1], Color_List[j % 5] + 'x', markersize=4)
                        plt.plot(p_gt[0], p_gt[1], Color_List[j % 5] + 'o', markersize=2)
                        plt.text(p_[0], p_[1], str(j + 1) + 'r : (' + str(round(r_new, 4)) + ')',
                                 size=8, alpha=0.9, color=Color_List[j % 5])

                    if l2_dis_t_1 < 32.0:
                        Agent.store_transition(s_input, a[0], r_new, s_input_)
                    else:
                        break

                print('Episode : [{0}] ---- Agent Pointer : [{1}]'.format(epoch_idx, Agent.pointer))
                print("\033[1;34;34mstep count : \033[0m", step_count)
                print('\nreward : [ {0} ]'.format(reward_list))
                print('var_list : [ {0} ]'.format(var_list))

                if plt_fig:
                    print('-----------------result display !!!')
                    plt.savefig(display_dir + 'result_' + str(train_step) + '_img_' + str(env.img_idx) + '.png',
                                bbox_inches='tight')

                if Agent.pointer > 1000:
                    print("\033[1;32;32mTraining step : [ {0} ] -- Agent pointer : [ {1} ]\033[0m".format(train_step,
                                                                                                         Agent.pointer))

                    # print('\n Training step : [ {0} ] -- Agent pointer : [ {1} ]'.format(train_step, Agent.pointer))
                    train_step += 1
                    summary = Agent.learn(train_step)
                    Agent.writer.add_summary(summary, train_step)
                    if train_step > 0 and train_step % 1000 == 0:
                        # save Actor Network weights every 1000 train_step
                        Agent.save_actor_net(step=train_step)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)