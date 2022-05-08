import os

if __name__ == '__main__':
    print("------------------Part 1：构建负例------------------")
    build_neg_sh = str('python build_neg_sample.py')
    a = os.system(build_neg_sh)
    if a == 0:
        print("开始构建负例图片。\n")
    else:
        print("运行build_neg_sample.py出错！\n")

    print("------------------Part 2：提取特征------------------")
    get_hog_sh = str('python get_hog.py')
    b = os.system(get_hog_sh)
    if b == 0:
        print("开始提取图片的Hog特征。\n")
    else:
        print("运行get_hog.py出错！\n")

    print("------------------Part 3：训练模型------------------")
    train_sh = str('python train.py')
    b = os.system(train_sh)
    if b == 0:
        print("开始训练SVM分类器。\n")
    else:
        print("运行train.py出错！\n")

    print("------------------Part 3：人脸检测------------------")
    detection_sh = str('python detection.py')
    b = os.system(detection_sh)
    if b == 0:
        print("开始进行人脸检测。\n")
    else:
        print("运行detection.py出错！\n")


