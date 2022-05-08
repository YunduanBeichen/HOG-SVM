import cv2
from tqdm import tqdm
import os


def mkdir(filepath):
    """
    创建文件夹
    :param filepath: 需要创建的目录，要求输入相对路径
    :return:
    """
    try:
        File_path = os.getcwd() + "\\" + filepath
        # print(File_path)
        if not os.path.exists(File_path):
            os.makedirs(File_path)
            print("新建目录成功：" + File_path)
        else:
            print("目录已经存在：" + File_path)
    except BaseException as e:
        print("新建目录失败：" + e)


def get_box(box_path, img_name):
    """
    获取图片人脸的边界框信息。
    :param box_path: bogndingbox目录paht
    :param img_name: 对应图片的名称
    :return: 返回一个包含四个int整数的列表，前两个为边框左上角坐标、后两个为右下角坐标
    """
    with open(box_path + "\\" + img_name[:-4] + '_boundingbox.txt', 'r', encoding='utf-8') as f:
        axis = f.readline()
    axis_list = axis.split(' ')
    # 删除最后一个空元素，防止float转换时报错。
    axis_list.pop()
    # print(axis_list)

    # 将字符串元素转换为整形元素
    axis_new_list = list(map(lambda x: int(float(x)), axis_list))
    # print(axis_new_list)
    return axis_new_list


def build_neg_img(original_folder, box_folder, output_folder):
    mkdir(output_folder)
    for filename in tqdm(os.listdir(original_folder)):
        # print(filename)
        original_img = cv2.imread(original_folder + "\\" + filename)
        axis_list = get_box(box_folder, filename)
        # 设边界框的左上角为M点，右下角为N点，图片的原点位于左上角，x轴向右为正方向，y轴向下为正方向
        Mx = axis_list[0]
        My = axis_list[1]
        Nx = axis_list[2]
        Ny = axis_list[3]
        box_width = Nx - Mx
        box_height = Ny - My

        # 与人脸边界框错开构建负例图像，左上角起始点为K
        if Mx - box_width >= 0:
            # 在边界框左侧框选同样大小的负例框
            Kx = Mx - box_width
            Ky = My
        elif My - box_height >= 0:
            # 在边界框上方选同样大小的负例框
            Kx = Mx
            Ky = My - box_height
        else:
            # 在整个图像的左上角选边界框的小的负例框
            Kx = original_img.shape[1] - box_width - 1
            Ky = 0

        # print(Kx, Ky)
        # print(box_width, box_height)

        neg_img = original_img[Ky: Ky + box_height, Kx: Kx + box_width, :]
        # cv2.imshow('neg_img', neg_img)
        # cv2.waitKey()
        cv2.imwrite(output_folder + "\\" + filename[:-4] + "_neg.jpg", cv2.resize(neg_img, (100, 100)))


if __name__ == '__main__':
    # 候选框路径、原始图片路径、切割图片路径
    boundingbox_path = r'face-detection\Annoatation\boundingbox'
    original_path = r'face-detection\Image\original'
    aligned_path = r'face-detection\Image\original'
    negative_path = r'face-detection\Image\negative'
    # print(os.getcwd())

    build_neg_img(original_path, boundingbox_path, negative_path)
    print("负例图片构建完成！")
