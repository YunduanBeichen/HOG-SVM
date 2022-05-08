import os
import cv2
import pickle
import numpy as np
from skimage import feature as ft
from tqdm import tqdm
from build_neg_sample import mkdir


def get_hog_skimage(img, orientations=10, pixels_per_cell=[12, 12], cells_per_block=[4, 4], visualize=False):
    """
    使用skimage库中的hog函数提取图像的hog特征，skimage库中的hog方法在cell级别没有进行高斯平滑。
    :param img:灰度图
    :param orientations:bin的个数
    :param pixels_per_cell:每个cell的大小
    :param cells_per_block:每个block中含有多少cell
    :param visualize:是否输出HOG image
    :return:visualize如果等于True时，可以绘制features[1]。如果等于False时，返回一维向量。
    """
    # 默认情况下，transform_sqrt=True，进行gamma correction，将较暗的区域变量，减少阴影和光照变化对图片的影响。
    features = ft.hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                      visualize=visualize, transform_sqrt=True)
    return features


def build_hog_data(folder_path):
    """
    获取对应文件夹下所有图片的hog特征
    :param folder_path: 文件夹的相对路径
    :return: 用于训练的train数据和用于测试的test数据
    """
    train_data = []
    test_data = []
    for filename in tqdm(os.listdir(folder_path)):
        img = cv2.imread(folder_path + "\\" + filename, 0)
        # print(img.shape)
        img_hog = get_hog_skimage(img)
        # print(img_hog)
        if filename.find('train') != -1:
            train_data.append(img_hog)
        else:
            test_data.append(img_hog)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    # print(train_data)
    return train_data, test_data


if __name__ == '__main__':
    pos_folder = r'face-detection\Image\aligned'
    neg_folder = r'face-detection\Image\negative'
    # pos_folder = r'fake\Image\aligned'
    # neg_folder = r'fake\Image\negative'
    data_pkl_folder = r'pkl\data'

    print("开始提取图片的Hog特征。\n")
    # 获取aligned文件夹下所有图片的hog特征，作为正例，numpy数组形式存储
    pos_train_data, pos_test_data = build_hog_data(pos_folder)
    pos_train_label = np.ones(pos_train_data.shape[0])
    pos_test_label = np.ones(pos_test_data.shape[0])
    print("正例图片特征提取完成！\n")

    # 获取negative文件夹下所有图片的hog特征，作为负例，numpy数组形式存储
    neg_train_data, neg_test_data = build_hog_data(neg_folder)
    neg_train_label = np.zeros(neg_train_data.shape[0])
    neg_test_label = np.zeros(neg_test_data.shape[0])
    print("负例图片特征提取完成！\n")

    # print(pos_train_data.shape, neg_train_data.shape)

    # 合并正负例训练、测试数据形成训练集、测试集
    X_train = np.concatenate((pos_train_data, neg_train_data), axis=0)
    Y_train = np.concatenate((pos_train_label, neg_train_label), axis=0)
    X_test = np.concatenate((pos_test_data, neg_test_data), axis=0)
    Y_test = np.concatenate((pos_test_label, neg_test_label), axis=0)

    # print(X_train.shape)

    # 保存为pkl文件，方便后续使用
    mkdir(data_pkl_folder)
    with open(data_pkl_folder + "\\X_train.pkl", 'wb') as f:
        pickle.dump(X_train, f)
    with open(data_pkl_folder + "\\Y_train.pkl", 'wb') as f:
        pickle.dump(Y_train, f)
    with open(data_pkl_folder + "\\X_test.pkl", 'wb') as f:
        pickle.dump(X_test, f)
    with open(data_pkl_folder + "\\Y_test.pkl", 'wb') as f:
        pickle.dump(Y_test, f)
    print("pkl文件保存完成！\n")
