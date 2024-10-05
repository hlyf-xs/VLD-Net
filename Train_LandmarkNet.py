from Landmark_Reg_1by1.Landmarknet_one_by_one import LandmarkNet
import os
from tqdm import tqdm
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('VB Detection', add_help=False)
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--IMG_SAVE_DIR', default='../dataset/vb_patch_dataset/TrainSet/aug_patch_img/', type=str)
    parser.add_argument('--Epoch', default=400, type=int)

    return argparse


def main(args):

    learning_rate = args.learning_rate
    batch_size = args.batch_size

    IMG_SAVE_DIR = args.IMG_SAVE_DIR
    img_dirs = os.listdir(IMG_SAVE_DIR)

    Epoch = args.Epoch
    img_num = len(img_dirs)
    batch_num = int(img_num / batch_size)


    train_step = 0
    per_epoch_train_step = int(img_num / batch_size)

    print('per_epoch_train_step: ', per_epoch_train_step)
    print('img num : {0}   batch size : {1}  batch num : {2}'.format(img_num, batch_size, batch_num))

    reg_net = LandmarkNet()

    for epoch in tqdm(range(Epoch)):

        if epoch % 100 == 0 and epoch > 0:
            reg_net.save_network_weights(step=epoch)

        for idx in tqdm(range(batch_num)):

            batch_input_tensor = reg_net.get_batch_img_tensor(batch_idx=idx)
            batch_landmark_tensor = reg_net.get_batch_gt_tensor(batch_idx=idx)

            summary = reg_net.learn(train_step=train_step,
                                    batch_input_tensor=batch_input_tensor,
                                    batch_gt_landmark=batch_landmark_tensor)

            train_step += 1
            reg_net.writer.add_summary(summary, train_step)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)