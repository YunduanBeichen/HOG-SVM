import os
import pickle
import cv2
from tqdm import tqdm
from get_hog import get_hog_skimage
from build_neg_sample import mkdir


def get_block(img, record):
    block = img[record[0]:record[0] + record[2], record[1]:record[1] + record[2]]
    return block


def sliding_window(img_name, img, classifier):
    # 设置初始窗长与原图高的比例为1:8
    scale = 4
    # 设置初始滑窗长度（正方形）和步长
    win_size = img.shape[0] // scale
    # 保证步长大于1
    step_size = max(win_size // 10, 1)

    candidate_box = []
    score = []
    while scale >= 2:
        # 设置窗的左上角为A点，初始化为（0，0）
        Ax = 0
        Ay = 0
        # 层优先滑窗
        while Ay + win_size < img.shape[0]:
            while Ax + win_size < img.shape[1]:
                win_hog = get_hog_skimage(cv2.resize(img[Ay:Ay + win_size, Ax:Ax + win_size], (100, 100))).reshape(
                    (1, -1))
                y_pred = classifier.predict(win_hog)
                decision = classifier.decision_function(win_hog)
                # print(decision)
                if y_pred == 1:
                    candidate_box.append((Ax, Ay, win_size))
                    score.append(decision)
                Ax += step_size
            Ax = 0
            Ay += step_size
        Ay = 0
        # result的结构是：每行表示一个候选框，每行的元素依次是左上角x坐标、左上角y坐标、窗长
        if len(candidate_box) > 0:
            break
        else:
            scale -= 1
            win_size = img.shape[0] // scale

    if len(candidate_box) == 0:
        print("\n" + img_name + "未检测到人脸\n")
    else:
        # print(len(candidate_box))
        # print(len(score))
        best_index = score.index(max(score))
        rel_img = cv2.rectangle(img, (candidate_box[best_index][0], candidate_box[best_index][1]), (
            candidate_box[best_index][0] + candidate_box[best_index][2],
            candidate_box[best_index][1] + candidate_box[best_index][2]),
                                (240, 124, 130),
                                2)
        cv2.imwrite(max_output_path + "\\" + filename[:-4] + "_max.jpg", rel_img)
        # cv2.imshow("candidate_box", rel_img)
        # cv2.waitKey(0)

        # 进行非极大值抑制(默认IoU阈值为0.2)
        result = nms(candidate_box, score)

        # 绘制候选框
        for best_index in range(len(result)):
            result_img = cv2.rectangle(img, (result[best_index][0], result[best_index][1]),
                                       (result[best_index][0] + result[best_index][2],
                                        result[best_index][1] + result[best_index][2]), (255, 0, 0), 2)
        cv2.imwrite(nms_output_path + "\\" + filename[:-4] + "_nms.jpg", result_img)
        # cv2.imshow("result", result_img)
        # cv2.waitKey(0)


def nms(box, evaluation, threshold=0.2):
    """
    对候选集进行非极大值抑制操作
    :param box: 候选集
    :param evaluation: 评价指标
    :param threshold: IoU阈值，默认为0.2
    :return: 非极大值抑制后的候选集
    """
    results = []
    while len(box) != 0:
        # step 1：根据置信度得分进行排序
        max_score = max(evaluation)
        max_index = evaluation.index(max_score)
        # step 2：选择置信度最高的候选框加入到最终结果列表，并在候选框中删除
        results.append(box[max_index])
        del box[max_index]
        del evaluation[max_index]
        # step 3：计算所有边界框的面积（但是由于同一scale下的面积相等，所以可以省略此步骤）
        # step 4：计算置信度最高的边框与其余边框的IoU
        box_temp = []  # 此处解决for循环list下表超出问题，具体可以参考：https://blog.csdn.net/weixin_43269020/article/details/88191630
        temp_score = []
        for index, value in enumerate(box):
            IoU = cal_IoU(results[-1], value)
            if IoU < threshold:
                box_temp.append(value)
                temp_score.append(evaluation[index])
        box = box_temp
        evaluation = temp_score
    return results


def cal_IoU(box1, box2):
    """
    两个边界框的交集部分除以它们的并集
    :param box1: 阈值最大的边框参数列表
    :param box2: 候选框参数列表
    :return: 二者的IoU
    """
    box1_area = box1[2] ** 2
    box2_area = box2[2] ** 2
    left_column_max = max(box1[0], box2[0])
    right_column_min = min(box1[0] + box1[2], box2[0] + box2[2])
    up_row_max = max(box1[1], box2[1])
    down_row_min = min(box1[1] + box1[2], box2[1] + box2[2])

    if left_column_max >= right_column_min or up_row_max >= down_row_min:
        return 0
    else:
        cross_area = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return cross_area / (box1_area + box2_area - cross_area)


if __name__ == '__main__':
    boundingbox_path = r'face-detection\Annoatation\boundingbox'
    original_path = r'face-detection\Image\original'
    aligned_path = r'face-detection\Image\original'
    negative_path = r'face-detection\Image\negative'
    max_output_path = r'output\max'
    nms_output_path = r'output\nms'
    model_pkl_folder = r'pkl\model'

    mkdir(max_output_path)
    mkdir(nms_output_path)

    # 读取svm模型
    with open(model_pkl_folder + "\\svm.model", 'rb') as f:
        clf = pickle.load(f)

    # 对图像进行识别
    for filename in tqdm(os.listdir(original_path)):
        image = cv2.imread(original_path + "\\" + filename, 0)
        sliding_window(filename, image, clf)
    print("检测图片保存完成！\n")
