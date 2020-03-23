# 决策树简介
决策树,是一项十分经典的机器学习的方法,是一类通过对训练数据进行划分而得到的树结构,通过对原有训练数据划分而对现有需要预测样例进行判断举个栗子,如果我们现在得到的西瓜数据集如下:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200226141422977.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xhdWdoX3hpYW9hbw==,size_16,color_FFFFFF,t_70)
其对应每一个样例,都有色泽,根蒂等特征,那么通过划分这些特征,我们就可以依据这个训练数据集生成对应的决策树:![在这里插入图片描述](https://img-blog.csdn.net/20180425192519753)
那么问题来了,如何去选择对应的特征来生成对应的决策树呢?

# 特征选择
对于决策树特征选择来说一般都是采用贪心的方法来进行选择特征,例如如上的训练数据集在每一次划分节点时,都按照“色泽->根蒂->敲声....”的**从左到右**的顺序去判断对应的划分节点.
#	划分指标
那么既然已经定了对应的特征的选择的次序,怎么评判对应划分后的好坏呢?
##	ID3决策树
这里我们引入一个**信息熵**的概念,用这个指标也就可以对于我们划分后的数据集的**纯度**进行评判(也就是我们数据集中属于同一类的数据有多少).公式如下:
![在这里插入图片描述](https://img-blog.csdn.net/20180425191239568)
因此,只要求划分后我们的数据集纯度越高,则表明我们的划分特征也就越好.自然而然的,划分前有一个信息熵,划分后又有一个信息熵,为了便于比较,我们把每一次划分使得数据集**纯度提升的多少**,称为**信息增益**,公式如下:
![](https://img-blog.csdn.net/20180425191454287)
对应的,我们还是举一个简单的栗子,还是使用原先的数据集:
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020022614455673.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xhdWdoX3hpYW9hbw==,size_16,color_FFFFFF,t_70)
首先我们先计算没有划分前的信息熵的值为多少,运用小学二年级一通运算后,我们得到如下的数值:
![在这里插入图片描述](https://img-blog.csdn.net/20180425191651329)
然后我们采用色泽先划分,因为色泽一共有三个**青绿，乌黑，浅白**三个特征,因此很自然的分成了三个部分,其好瓜样本占属性样本比例分别是3/6，4/6，1/5。坏瓜样本占属性样本比例分别是3/6，2/6，4/5。则根据信息熵公式得到：![在这里插入图片描述](https://img-blog.csdn.net/20180425191757987)
然后再根据信息增益的公式可以得到:
![](https://img-blog.csdn.net/20180425191808143)
同理,我们还需要对根蒂,敲声等特征再进行计算,最后留下对应的信息增益最大的方法.
![在这里插入图片描述](https://img-blog.csdn.net/2018042519182174)
很容易可以看到最大的就是纹理,因此我们采用纹理作为我们最后的划分特征,从而得到:
![](https://img-blog.csdn.net/20180425191909367)
同理只要一直这样划分,就可以得到对应的最后的决策树啦,需要注意的是,之前采用过的划分特征,在其子节点的划分过程中,是不可以采用的哦.
## C4.5决策树
由于权重的存在，信息增益其实对包含数目较多的属性有偏好。为了减少这种不“客观”的判定，可以选择”增益率“(C4.5)来划分属性,其实说白了也就是对信息增益除以一个属性的数目,从而减小对应的影响:
![](https://img-blog.csdn.net/2018042519220894)
IV(a)计算的其实就是每一个划分后的数据集的大致比例.
![在这里插入图片描述](https://img-blog.csdn.net/20180425192218386)
## CART决策树
当然了,你也可以采用另外指标如基尼指数来对树进行划分.具体的公式如下:
![在这里插入图片描述](https://img-blog.csdn.net/20180425191559497)
![在这里插入图片描述](https://img-blog.csdn.net/20180425191608671)
基尼指数,主要是对同一个集合中,不同样例多少,即非纯度的一个评价,跟信息熵正好相反.其他的划分过程则依旧不变.
总结来说,目前绝大多数的决策树采用的CART的决策树,而且效果一般要比ID3和C4.5的决策树要好.

 ## 决策树剪枝
 由于决策树是对训练数据集直接划分得到的,因此十分容易造成过拟合的情况,所以我们还需要对决策树进行适当的剪枝来减小模型的复杂度,
 ### 预剪枝
预剪枝的思想很简单,即当我们采用的指标没有获得较大的提升,就不在往下划分树了.举一个简单的栗子,例如对于当前划分情况:
![在这里插入图片描述](https://img-blog.csdn.net/20180906110327911?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pmYW41MjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
当我们计算出来对应根蒂的指标不如不划分之前的指标大小,那么我们就停止划分,进而得到对应的决策树.
![在这里插入图片描述](https://img-blog.csdn.net/20180906110951766?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pmYW41MjA=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
### 后剪枝
后剪枝同预剪枝相反,其在整棵树生成完之后,通过对整棵树进行遍历,挑选那些划分后信息增益反而降低的分支进行裁剪,从而降低整棵树的过拟合情况.

# 代码实战
在代码实战这块,我采用的是sklearn的框架进行的快速搭建,最后对决策树分类的内容进行了一个简单的可视化,具体代码如下:
```python
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
```
最后跑出来的结果哦~:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200228182409310.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xhdWdoX3hpYW9hbw==,size_16,color_FFFFFF,t_70)
除此之外,我还添加了对应的可视化部分:
```python
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
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200228182419907.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0xhdWdoX3hpYW9hbw==,size_16,color_FFFFFF,t_70)
# 总结
限于篇幅的原因,只是简单的介绍了决策树的大概内容,至于对应的决策树如缺失值的处理,预剪枝后剪枝的优缺点等都没有详细概要的介绍,期望大家可以自己学习一下啦.

# 参考文献
[西瓜书学习（一）—决策树（上）](https://blog.csdn.net/quinn1994/article/details/80083933)
[决策树的预剪枝与后剪枝](https://blog.csdn.net/zfan520/article/details/82454814)
[利用plt.pcolormesh绘制分类图](https://blog.csdn.net/zsdust/article/details/79726118)