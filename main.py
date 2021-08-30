from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba


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

def dict_demo():
    data_dict = [{"City": "Beijing", "Temperature": 20},
                 {"City": "Shanghai", "Temperature": 30},
                 {"City": "Shenzhen", "Temperature": 25}]
    transfer = DictVectorizer(sparse=False)
    # 默认返回稀疏矩阵,省略0的存储空间 sparse=False来转换为普通矩阵
    new_data = transfer.fit_transform(data_dict)
    print(data_dict)
    print(new_data)
    print("特征名字为:", transfer.get_feature_names())
    return None


def divide():
    text = "我是你的父亲"
    text = " ".join(list(jieba.cut(text)))
    print(text)
    print(type(text))


def txt_demo():  # 单词作为特征
    # test_txt = ["life is in your hand, and so do the tomorrow ,which is also in your hand!"]
    test_txt = ["我爱在北京邮电大学里面坐牢", "乔建永也在里面坐牢"]

    transfer = CountVectorizer(stop_words=["里面", "在里面"])
    new_txt = transfer.fit_transform(test_txt)
    print("处理后的数据为:\n", new_txt)
    print("处理后的数据为:\n", new_txt.toarray())
    print("处理后的数据特征名称为:\n", transfer.get_feature_names())
    return None


if __name__ == '__main__':
    # load()
    # dict_demo()
    # txt_demo()
    divide()
