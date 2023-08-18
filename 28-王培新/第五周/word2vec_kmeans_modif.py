#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path): #尽量用大点的语料来训练词向量。 然后加载磁向量。
    model = Word2Vec.load(path)
    return model

#新增资金入场 沪胶强势创年内新高
def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence))) ##将分词用空格连接起来
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences: ##遍历文本中的每一句话，
        words = sentence.split()  #sentence是分好词的，空格分开，先做分词。
        vector = np.zeros(model.vector_size) ##以全0为初始向量。
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，在词向量里也就没有，使用全0向量代替  ；
                # 如果做的细点可以再训练中准备unk向量，把所有的低频词当做unk;做粗点就把他忽略了。
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

#--------------------------------------add---------------start------------------------------
def __distance( p1, p2):
    #计算两点间距，欧式距离
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)
def __sumdis_all(vectors,labels,cluster_centers_):
    ##sentences  vectors labesl 一一对应
    # label就是cluster_centers的下标
    #计算总距离和  ，遍历质点，遍历质点所在的簇的点的  距离总和
    vectors_label_dict = defaultdict(list)  ##类似于自己实现的reslut [[],[],[]......];
    for vector, label in zip(vectors, labels):  #取出句子和标签
        vectors_label_dict[label].append(vector)         #同标签的句子放到一起

    sum=[0]*(len(cluster_centers_))

    for i in range(len(cluster_centers_)):
        sum_d=0
        for j in range(len(vectors_label_dict[i])):
            sum_d+=__distance(vectors_label_dict[i][j],cluster_centers_[i])
        sum[i]=sum_d/(len(vectors_label_dict[i])+1)
    return sum  ##每个质心所在簇的类内平均距离；
#--------------------------------------add---------------end------------------------------

def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化，10000条标题，就转换成了10000*100的矩阵。

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量 100条
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类  ##三方库封装好的，sklearn里面，初始化的时候指定聚类数量
    kmeans.fit(vectors)          #进行聚类计算 ，训练时候就把所有的矩阵送进去，调用fit函数。

    sentence_label_dict = defaultdict(list)  ##类似于自己实现的reslut [[],[],[]......];
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的句子放到一起
#-------------------------------add ------------------start----------------------------

    sum_all=__sumdis_all(vectors, kmeans.labels_,kmeans.cluster_centers_)
    sum_dict={index:x for index,x in enumerate(sum_all)}
    sum_all = sorted(sum_dict.items(), key=lambda x: x[1], reverse=False) ##升序  元素为元组的列表

    ##打印类内平均距离最小的前10个类的数据
    sentence_label_dict_ten = { key:sentence_label_dict[key] for index,(key,value)  in enumerate(sum_all) if index<10}
    for label, sentences in sentence_label_dict_ten.items(): ##按照标签再把结果打印出来，没有借助标注，完成了文本聚类的结果，在nlp中无监督的任务相对较少
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))  ##将空格替换成空；
        print("---------")
#-------------------------------add -----------------end  ----------------------------

if __name__ == "__main__":
    main()

