## CVLab3：Silding Window代码运行说明

### 文件结构

#### CVLab3：整个工程的根目录，包括了face-detection、output、pkl三个文件夹，build_neg_sample.py、detection.py、get_hog.py、train.py四个Part的python文件，和一体化的main.py文件

###### face-detection目录：

数据集目录，存放提供RAF-DB，以及build_neg_sample.py生成的负例图片。

face-detection\Image\negative 为负例图片输出路径。

###### output目录：

输出目录，存放detection.py人脸检测的结果。

output\max 使用候选框最大值方法的结果。

output\nms 使用非极大值抑制方法的结果。

###### pkl目录：

训练集、模型保存目录，存放get_hog.py生成的训练集以及train.py得到的SVM分类器。

pkl\data 存放提取hog特征的训练集、测试集

pkl\model 存放训练的SVM模型

###### build_neg_sample.py：

用于构建负例图片数据集。

###### get_hog.py：

用于提取图片的Hog特征。

###### train.py：

用于训练SVM分类器。

###### detection.py：

用于人脸检测，绘制边界框。

###### main.py：

整体运行文件。

### 运行说明

将数据集解压到根目录下后，运行main.py文件即可。

```shell
python main.py
```

若需要单独运行某一环节，可直接运行对应python文件。