from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import feature_extraction
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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


def divide(m_input):
    text = m_input
    text = " ".join(list(jieba.cut(text)))
    print(text)
    #  print(type(text))
    return text


def divide2(input_1):
    my_data = input_1
    my_data_0 = []
    for sentence in my_data:
        my_data_0.append(divide(sentence))
    print(my_data_0)
    return my_data_0


def txt_demo(input_0):  # 单词作为特征
    # test_txt = ["life is in your hand, and so do the tomorrow ,which is also in your hand!"]
    test_txt = divide2(input_0)
    transfer = CountVectorizer(stop_words=["里面", "在里面"])
    new_txt = transfer.fit_transform(test_txt)
    print("处理后的数据为:\n", new_txt)
    print("处理后的数据为:\n", new_txt.toarray())
    print("处理后的数据特征名称为:\n", transfer.get_feature_names())
    return None


def tfidf_demo(input_2):
    test_txt = divide2(input_2)
    transfer = TfidfVectorizer()
    new_txt = transfer.fit_transform(test_txt)
    print("处理后的数据为:\n", new_txt)
    print("处理后的数据为:\n", new_txt.toarray())
    print("处理后的数据特征名称为:\n", transfer.get_feature_names())
    return None


if __name__ == '__main__':
    # load()
    # dict_demo()
    # txt_demo()
    my_data_outside = ["如果拿望远镜去看别人，就会拿放大镜来看自己。",
                       "读不在三更五鼓，功只怕一曝十寒。",
                       "发光并非太阳的专利，你也可以发光。",
                       "世上没有绝望的处境，只有对处境绝望的人。",
                       "把活着的每一天看作生命的最后一天。"]
    tfidf_demo(my_data_outside)
