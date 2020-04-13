"""numpy实现简单神经网络"""
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

def Cross_entropy(x, y):
    # 交叉熵损失函数
    loss = -np.log(x) * y
    n = len(loss)
    return sum(np.sum(loss, axis=1) / n)  # 行求和


def softmax(x):
    x = np.exp(x)
    sumval = np.reshape(np.sum(x, axis=1), (x.shape[0], 1))
    return x / sumval


def sigmoid(x):
    # sigmoid激活函数
    return 1 / (1 + np.exp(-x))


def Derivative_sigmoid(x):
    # sigmoid函数的导数
    return np.matmul(sigmoid(x), (1 - sigmoid(x)).T)


def Accuracy(prediction, labels):
    prediction = np.argmax(prediction, axis=1)
    labels = np.argmax(labels, axis=1)
    counter = Counter(prediction + labels)
    return counter[2] / len(prediction)


if __name__ == '__main__':
    # 对读取的数据进行预处理.
    iris = pd.read_csv("iris.csv", index_col=0)
    data, labels = np.array(iris.iloc[:, :4]), iris.iloc[:, -1]

    # 将标签转化成对应的onehot编码。
    onehot_encoder = OneHotEncoder()

    # 将文字转换成对应的数字
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = np.reshape(labels, (-1, 1))
    labels = onehot_encoder.fit_transform(labels).toarray()

    # hidden_size为隐藏神经元数目，output_size为输出的类别数
    hidden_size, output_size = 100, labels.shape[1]

    # 根据输入得到对应的输入维度
    input_size = len(data[0])

    # 构建隐藏层权重，shape为(I，O)
    weight = np.random.random_sample((input_size, output_size))

    # 设置超参数轮次Epoch和Batch_size
    Epoch, Batch_Size, learning_rate = 1, 32, 0.01

    # 分割数据集，留下20%的数据来预测模型的结果.
    traindata, testdata, trainlabels, testlabels = train_test_split(data, labels, test_size=0.2)
    for e in range(Epoch):
        # 训练Epoch个轮次
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2020)  # 五折交叉验证
        for train, valid in kfold.split(data, [0] * len(labels)):
            traindata, trainlabels, validdata, validlabels = data[train], labels[train], data[valid], labels[valid]
            loss = 0
            for b in range(0, len(traindata), Batch_Size):
                batch_data, batch_labels = traindata[b:b + Batch_Size], trainlabels[b:b + Batch_Size]

                # output_data维度应该为（B，H）*(H, O)=(B, O)
                output_data = softmax(np.matmul(batch_data, weight))

                # 计算对应的交叉熵损失函数
                loss += Cross_entropy(output_data, batch_labels)

                # Gradient Descent 梯度下降更新, 首先更新输出层的权重.
                output_layer_delta = (batch_labels - output_data).T
                output_w_delta = np.matmul(output_layer_delta, batch_data) / Batch_Size
                weight += learning_rate * output_w_delta.T
            print(loss)

        # 最后预测的部分
        output_data = softmax(np.matmul(testdata, weight))
        print("正确率", Accuracy(output_data, testlabels))
