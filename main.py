from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction
def load():
    iris = load_iris()
    #  print("鸢尾花数据集的描述为:\n", iris["DESCR"])
    x_train, x_test, y_train, y_size = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值为:\n", x_train)
    print("训练集的特征值数目为:\n", x_train.shape)
    print("鸢尾花的数据集为:\n", iris.data.shape)
    return None


# 　返回值为字典数据类型,共有五个键 data:特征数据数组 target:标签数组 DESCR:数据描述
#   feature_names:特征名 target_names:标签名

if __name__ == '__main__':
    load()
