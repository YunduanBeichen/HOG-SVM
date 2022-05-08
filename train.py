import pickle
import numpy as np
from sklearn.svm import SVC
from build_neg_sample import mkdir

if __name__ == '__main__':
    X_train_pkl = r'pkl\data\X_train.pkl'
    Y_train_pkl = r'pkl\data\Y_train.pkl'
    X_test_pkl = r'pkl\data\X_test.pkl'
    Y_test_pkl = r'pkl\data\Y_test.pkl'
    model_pkl_folder = r'pkl\model'

    # 加载训练集
    with open(X_train_pkl, 'rb') as f:
        X_train = pickle.load(f)
    # print(X_train.shape)
    with open(Y_train_pkl, 'rb') as f:
        Y_train = pickle.load(f)
    # print(Y_train.shape)
    with open(X_test_pkl, 'rb') as f:
        X_test = pickle.load(f)
    # print(X_test.shape)
    with open(Y_test_pkl, 'rb') as f:
        Y_test = pickle.load(f)
    # print(Y_test.shape)
    print("加载训练集完成！\n")

    # 训练svm
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    true = np.sum(y_pred == Y_test)
    print("预测正确的数目为：", true)
    print('预测错的的结果数目为：', Y_test.shape[0] - true)
    print('预测结果准确率为：', true / Y_test.shape[0])
    # print("准确率为：", classifier.score(X_train, Y_train))
    # print(classifier.decision_function(X_train))

    # 保存svm模型
    mkdir(model_pkl_folder)
    with open(model_pkl_folder + "\\svm.model", 'wb+') as f:
        pickle.dump(classifier, f)
        print("SVM模型保存完成!")
        f.close()
