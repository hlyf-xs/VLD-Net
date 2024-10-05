from Landmark_Reg_1by1.Landmarknet_one_by_one import LandmarkNet
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


model_dir = '../Landmark_Reg_1by1/model_weights/'

istrain = False
isTest = True
isVal = False

if istrain:
    img_dir = '../dataset/att_region_dataset/Train/img/'
    dataset = 'TrainSet'
elif isTest:
    img_dir = '../dataset/att_region_dataset/Test/img/'
    dataset = 'TestSet'
else:
    img_dir = '../dataset/att_region_dataset/Val/img/'
    dataset = 'ValSet'

img_num = len(os.listdir(img_dir))
print("img_num:", img_num)


def load_ori_img(index):
    ORI_IMG_SAVE_DIR = '../dataset/att_region_dataset/Val/img_with_att/'
    img_dirs = os.listdir(ORI_IMG_SAVE_DIR)
    global image_name
    image_name = img_dirs[index]
    img_path_name = os.path.join(ORI_IMG_SAVE_DIR, img_dirs[index])

    ori_img = cv2.imread(img_path_name)

    ORI_GT_SAVE_DIR = '../dataset/att_region_dataset/Val/GT_landmark/'
    gt_landmark_name = img_dirs[index].split(".")[0]
    gt_path_name = os.path.join(ORI_GT_SAVE_DIR, gt_landmark_name) + '.npy'
    ori_gt_landmark = np.load(gt_path_name)


    return ori_img, ori_gt_landmark


def load_patch_img(image_name, index):

    # cal image!
    PATCH_IMG_SAVE_DIR = '../dataset/vb_patch_dataset/' + dataset + '/patch_img/'
    patch_image_name = image_name[:-5] + "_" + str(index) + ".jpg"

    img_path_name = os.path.join(PATCH_IMG_SAVE_DIR, patch_image_name)
    patch_img = cv2.imread(img_path_name)

    PATH_GT_SAVE_DIR = '../dataset/vb_patch_dataset/' + dataset + '/GT_landmark/'
    gt_landmark_name = patch_image_name.split(".")[0]
    gt_path_name = os.path.join(PATH_GT_SAVE_DIR, gt_landmark_name) + '.npy'
    patch_gt_landmark = np.load(gt_path_name)

    return patch_img, patch_gt_landmark


landmark_net = LandmarkNet(out_graph=False)

landmark_net.load_network_weights(model_dir, step=300)


def load_center_point(pts):
    pts = np.asarray(pts, np.float32)  # 68 x 2
    num_pts = pts.shape[0]  # number of points, 68

    center_pts = []
    for i in range(0, num_pts, 4):
        pt1 = (pts[i, :] + pts[i + 2, :]) / 2
        pt2 = (pts[i + 1, :] + pts[i + 3, :]) / 2

        center_pt = (pt1 + pt2) / 2
        center_pts.append(center_pt)

    center_pts = np.array(center_pts)

    return center_pts


def trans_coords(pre_landmark_array, ori_center_pt):
    x = int(ori_center_pt[0])
    y = int(ori_center_pt[1])

    trans_coords = []

    if x < 50:
        left_top_x = pre_landmark_array[0][0] * 100
        left_dwon_x = pre_landmark_array[1][0] * 100
        right_top_x = pre_landmark_array[2][0] * 100
        right_down_x = pre_landmark_array[3][0] * 100
    elif x > 200:
        left_top_x = 150 + pre_landmark_array[0][0] * 100
        left_dwon_x = 150 + pre_landmark_array[1][0] * 100
        right_top_x = 150 + pre_landmark_array[2][0] * 100
        right_down_x = 150 + pre_landmark_array[3][0] * 100
    else:
        left_top_x = x + (pre_landmark_array[0][0] * 100 - 50)
        left_dwon_x = x + (pre_landmark_array[1][0] * 100 - 50)
        right_top_x = x + (pre_landmark_array[2][0] * 100 - 50)
        right_down_x = x + (pre_landmark_array[3][0] * 100 - 50)


    if y < 50:
        left_top_y = pre_landmark_array[0][1] * 100
        left_dwon_y = pre_landmark_array[1][1] * 100
        right_top_y = pre_landmark_array[2][1] * 100
        right_down_y = pre_landmark_array[3][1] * 100
    elif y > 700:
        left_top_y = 650 + pre_landmark_array[0][1] * 100
        left_dwon_y = 650 + pre_landmark_array[1][1] * 100
        right_top_y = 650 + pre_landmark_array[2][1] * 100
        right_down_y = 650 + pre_landmark_array[3][1] * 100
    else:
        left_top_y = y + (pre_landmark_array[0][1] * 100 - 50)
        left_dwon_y = y + (pre_landmark_array[1][1] * 100 - 50)
        right_top_y = y + (pre_landmark_array[2][1] * 100 - 50)
        right_down_y = y + (pre_landmark_array[3][1] * 100 - 50)

    trans_coords.append([left_top_x, left_top_y])
    trans_coords.append([left_dwon_x, left_dwon_y])
    trans_coords.append([right_top_x, right_top_y])
    trans_coords.append([right_down_x, right_down_y])

    trans_coords = np.array(trans_coords)

    return trans_coords


def draw_patch_landmarks(image, corrds, color=(0, 255, 255)):

    cv2.circle(image, center=(corrds[0, 0], corrds[0, 1]), radius=1, color=color,
               thickness=2)
    cv2.circle(image, center=(corrds[1, 0], corrds[1, 1]), radius=1, color=color,
               thickness=2)
    cv2.circle(image, center=(corrds[2, 0], corrds[2, 1]), radius=1, color=color,
               thickness=2)
    cv2.circle(image, center=(corrds[3, 0], corrds[3, 1]), radius=1, color=color,
               thickness=2)


def draw_ori_landmarks(pts, img, color=(255, 0, 255)):

    pts = np.asarray(pts, np.float32)   # 68 x 2

    num_pts = pts.shape[0]   # number of points, 68

    for i in range(0, num_pts, 4):

        cv2.circle(img, center=(int(pts[i, 0]), int(pts[i, 1])), radius=1, color=color, thickness=2)
        cv2.circle(img, center=(int(pts[i+1, 0]), int(pts[i+1, 1])), radius=1, color=color, thickness=2)
        cv2.circle(img, center=(int(pts[i+2, 0]), int(pts[i+2, 1])), radius=1, color=color, thickness=2)
        cv2.circle(img, center=(int(pts[i+3, 0]), int(pts[i+3, 1])), radius=1, color=color, thickness=2)


if __name__ == '__main__':

    print('img_num: ', img_num)
    for i in range(img_num):
        # print("i:", i)
        ori_img, ori_gt_landmark = load_ori_img(i)
        ori_center_pts = load_center_point(ori_gt_landmark)

        draw_ori_landmarks(ori_gt_landmark, ori_img, color=(0, 0, 255))

        pre_trans_landmarks = []
        for j in range(0, 17):
            img, gt_landmark = load_patch_img(image_name, j)

            Input_tensor = img[:, :, 0:1]
            Input_tensor = Input_tensor.astype('float32') / 255.0  # (100, 100, 1) float32

            output_tensor = landmark_net.pre_landmarknet(Input_tensor)
            pre_patch_landmarks = output_tensor[0]

            pre_patch_landmark_array = []
            pre_patch_landmark_array.append([pre_patch_landmarks[0], pre_patch_landmarks[4]])
            pre_patch_landmark_array.append([pre_patch_landmarks[1], pre_patch_landmarks[5]])
            pre_patch_landmark_array.append([pre_patch_landmarks[2], pre_patch_landmarks[6]])
            pre_patch_landmark_array.append([pre_patch_landmarks[3], pre_patch_landmarks[7]])


            pre_patch_landmark_array = np.array(pre_patch_landmark_array)

            ori_center_pt = ori_center_pts[j]

            trans_patch_coords = trans_coords(pre_patch_landmark_array, ori_center_pt)

            for h in range(0, 4):
                pre_trans_landmarks.append([trans_patch_coords[h][0], trans_patch_coords[h][1]])

            trans_patch_coords = trans_patch_coords.astype(int)

            pre_patch_landmark_array = (pre_patch_landmark_array*100).astype(int)
            gt_landmark = gt_landmark.astype(int)

            cv2.circle(ori_img, center=(int(ori_center_pt[0]), int(ori_center_pt[1])), radius=1, color=(0, 255, 255),
                       thickness=2)

            draw_patch_landmarks(ori_img, trans_patch_coords, color=(255, 0, 0))

            draw_patch_landmarks(img, gt_landmark, color=(0, 0, 255))
            draw_patch_landmarks(img, pre_patch_landmark_array, color=(255, 0, 0))

            show_data = False
            if show_data:
                plt.figure(i)
                plt.imshow(img)
                plt.plot(gt_landmark[:, 0], gt_landmark[:, 1], 'bo', markersize=3)
                plt.plot(pre_patch_landmark_array[:, 0] * 100, pre_patch_landmark_array[:, 1] * 100, 'ro', markersize=3)

                plt.show()

        pre_landmark_npy = np.array(pre_trans_landmarks)

        print("pre_landmark_npy:", pre_landmark_npy)