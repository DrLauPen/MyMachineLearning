import lightgbm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    iris = pd.read_csv("iris.csv", index_col=0)
    # 验证集合划分.
    data, labels = np.array(iris.iloc[:, :-1]), iris.iloc[:, -1]
    # 标签转为数字
    labelec = LabelEncoder()
    labels = labelec.fit_transform(labels)
    traindata, testdata, trainlabels, testlabels = train_test_split(data, labels, test_size=0.1)

    result = np.zeros((len(testlabels), 3))  # 提前设置一个维度为[15,3]的矩阵用于保留最后的预测结果.
    kfold = KFold(n_splits=5, shuffle=True, random_state=2020)
    model = lightgbm.LGBMClassifier(
        max_depth=5,
        num_leaves=25,
        learning_rate=0.7,
        n_estimators=100,
        min_child_samples=80,
        colsample_bytree=1,
    )

    for tra, val in kfold.split(traindata, trainlabels):
        model.fit(traindata[tra], trainlabels[tra])
        prediction = model.predict(traindata[val])

        # 计算该折模型下的正确率.
        print("Acc:{}".format(accuracy_score(prediction, trainlabels[val])))

        # 多折交叉运算保留每一折其中的一部分，也算一种集成的方式
        pre = model.predict_proba(testdata)  # 这里得到对应的概率值.
        result += pre / 5

    # 查找对应行下值的最大下标
    result = np.argmax(result, axis=1)
    print("Acc:{}".format(accuracy_score(result, testlabels)))
