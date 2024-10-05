import numpy as np
import matplotlib.pyplot as plt
from Environment_Scale_Size import Environment
from VB_Loc_Agent import Agent
import os


#  -----------------   Agent   --------------------------------
env = Environment()

Agent = Agent(out_graph=False)
MODEL_SAVE_PATH = 'models/'
Load_model_step = 110000
Agent.load_actor_net(model_dir=MODEL_SAVE_PATH, step=Load_model_step)


test_img_dir = "data/test_images"
test_img_list = os.listdir(test_img_dir)

img_num = len(test_img_list)

for img_idx in range(0, img_num):
    p, s = env.reset(img_idx=img_idx)

    pre_center_p = np.zeros((17, 2), dtype='float32')
    pre_center_p[0, :] = p

    # ******************************start VB Localization ******************************************
    for step_idx in range(17):

        s_input = env.make_s_input(s)
        a = Agent.choose_action(s_input)
        a_norm = np.random.normal(loc=a, scale=0.0)
        a = np.clip(a_norm, -1, 1)
        p_, s_ = env.step(a, p)
        pre_center_p[step_idx + 1, :] = p_
        s_input_ = env.make_s_input(s_)
        p, s = p_, s_
    # ******************************End VB Localization **********************************

    # ****************************** att region ******************************************
    _, ori_img = env.data_process.load_img(env.img_idx)
    ori_img_name = env.data_process.get_img_name(env.img_idx)
    print("img_idx, ori_img_name:", img_idx, ori_img_name)

    
    # ****************  plt spine  ********
    plt_fig = True
    if plt_fig:
        plt.close()
        plt.figure(env.img_idx)
        plt.imshow(ori_img)
        plt.plot(pre_center_p[:, 0], pre_center_p[:, 1], 'bo', markersize=4)

        plt.show()
    # ****************  plt spine  ********
