python Train_VB_localization.py --MAX_EPISODES=110000 --MAX_EP_STEPS=17 --per_img_explore_num=20 --out_graph=True -- train_img_dir='data/images' --display='display/'
python Train_LandmarkNet.py --learning_rate=0.0005 --batch_size=20 --IMG_SAVE_DIR='../dataset/vb_patch_dataset/TrainSet/aug_patch_img/' --Epoch=400