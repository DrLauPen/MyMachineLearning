import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    # 加载数据集iris,index_col是选取对应列做索引
    iris = pd.read_csv("/Users/XYJ/Downloads/Iris数据集/iris.csv", index_col="Unnamed: 0")

    # 转换成对应的numpy数组
    iris = np.array(iris)

    # 划分对应的训练数据和标签
    data, labels = iris[:, :4], iris[:, 4]
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    # 划分成训练集合和验证集合,默认采用二八划分
    traindata, testdata, trainlabels, testlabels = train_test_split(data, labels, test_size=0.2)

    # 构建二叉树,对应划分标准采用信息熵，集ID3算法
    decisionTree = tree.DecisionTreeClassifier(criterion='entropy')

    # 五折交叉验证
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    for train, valid in kfold.split(traindata, trainlabels):
        decisionTree.fit(traindata[train], trainlabels[train])

        result = decisionTree.predict(traindata[valid])
        # 计算准确率
        acc = accuracy_score(result, trainlabels[valid])
        print("正确率:{}".format(acc))
        
    # # 查看对应的特征重要性
    # print(decisionTree.feature_importances_)

    # 最后计算测试集合对应的结果
    result = decisionTree.predict(testdata)

    # 取首俩个特征画画图
    x1_min, x1_max = min(traindata[:, 0]), max(traindata[:, 0])  # 取对应列的最大最小值
    x2_min, x2_max = min(traindata[:, 1]), max(traindata[:, 1])
    # 从最小到最大，取500个点
    t1, t2 = np.linspace(x1_min, x1_max, 500), np.linspace(x2_min, x2_max, 500)
    coord_x, coord_y = np.meshgrid(t1, t2)  # 对应这些点组成的坐标

    #.flat首先将其降维成一维，再用np.stack按列叠加维度
    x_train= np.stack((coord_x.flat, coord_y.flat), axis=1)

    #用前两个属性来构造模型
    decisionTree.fit(traindata[:, :2], trainlabels)

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

    #设置对应的坐标轴的最大最小值
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    #按照刚才拼接回来的再进行预测
    y_hat = decisionTree.predict(x_train)

    #绘制面积图
    plt.pcolormesh(coord_x, coord_y, y_hat.reshape(coord_x.shape), cmap=cm_light)

    #绘制散点图
    plt.scatter(traindata[:, 0], traindata[:, 1], c=trainlabels, cmap=cm_dark, marker='o', edgecolors='k')
    #标题名字
    plt.title('Decision Tree')
    plt.show()
