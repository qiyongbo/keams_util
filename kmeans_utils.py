#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import pandas as pd

colors = ["b", "violet", "y", "g", "k"]
markers = [".", "s", "*", "v", "o"]
data_s = 20





def loadDataSet(fileName, func=lambda a: a, sliper=","):
    """
    获取数据
    fileName:文件名
    func:数据转换方式,参数为list,返回值也为list
    """
    numFeat = len(open(fileName).readline().split(',')) - 1  # get number of fields
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        if '\xef\xbb\xbf' in line:
            line = line.replace('\xef\xbb\xbf', '')
        curLine = line.strip().split(sliper)

        dataMat.append(func(curLine))
    return dataMat


def dataFilter(dataMat, descr, func):
    """
    筛选数据
    dataMat 源数据
    descr 描述
    func 筛选方法 输入为list 返回为布尔类型,True保留数据,False删除数据
    """
    o = len(dataMat)
    print "对数据进行 %s 规则的筛选,筛选之前数量为:%d" % (descr, o)
    d = [data for data in dataMat if func(data)]
    n = len(d)
    print "筛选之后,数量为:%d,删除数据量为:%d" % (n, o - n)
    return d


def loadLable(fileName):
    numFeat = len(open(fileName).readline().split(',')) - 1  # get number of fields
    fr = open(fileName)
    lable = []
    for line in fr.readlines():
        if line != "\n":
            curLine = line.split("\t")
            lable.append(curLine[1])
    return lable


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


def transOrig(normDataSet, ranges, minVals):
    m = normDataSet.shape[0]
    OrigDataSet = normDataSet * np.tile(ranges, (m, 1))
    OrigDataSet = OrigDataSet + np.tile(minVals, (m, 1))
    return OrigDataSet

def scaleNorm(dataSet):

    scale = np.std(dataSet, 0)
    mean = np.mean(dataSet, 0)
    m = dataSet.shape[0]
    scaleDataSet = dataSet - np.tile(mean, (m, 1))
    scaleDataSet = scaleDataSet / np.tile(scale, (m, 1))  # element wise divide
    return scaleDataSet, scale, mean

def scaleToOrig(scaleDataSet,scale, mean):
    m = scaleDataSet.shape[0]
    OrigDataSet = scaleDataSet * np.tile(scale, (m, 1))
    OrigDataSet = OrigDataSet + np.tile(mean, (m, 1))
    return OrigDataSet

def getColerMarker(index):
    c_i = index % len(colors)
    marker_i = index / len(colors)
    return c_i, marker_i


def plotKmean2D(Xs, Ys, Xlable, Ylable, centers, num):
    for i in range(0, num):
        c_i, marker_i = getColerMarker(i)
        plt.scatter(Xs[i], Ys[i], c=colors[c_i], marker=markers[marker_i], s=data_s, alpha=0.5)

    if Xlable is not None:
        plt.xlabel(Xlable)
        plt.ylabel(Ylable)

    if Ylable=="regi_days":
        max_days = 3650
        years = range(0,10)
        days = [y*365 for y in years]
        plt.yticks(days,years)
        plt.ylabel('regi_years')
    plt.scatter(centers[:, 0], centers[:, 1], c="r", marker="x", s=200)
    plt.show()


def plot3D(Xs, Ys, Zs, Xlable, Ylable, Zlable, centers, num):
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #  将数据点分成三部分画，在颜色上有区分度
    # ax.scatter(x[:10], y[:10], z[:10], c='y')  # 绘制数据点
    for i in range(0, num):
        c_i, marker_i = getColerMarker(i)
        ax.scatter(Xs[i], Ys[i], Zs[i], c=colors[c_i], marker=markers[marker_i], s=data_s, alpha=0.5)
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c="r", marker="x", s=100)
    # ax.scatter(x[30:40], y[30:40], z[30:40], c='g')

    ax.set_xlabel(Xlable)  # 坐标轴
    ax.set_ylabel(Ylable)
    ax.set_zlabel(Zlable)

    if Ylable=="regi_days":
        max_days = 3650
        years = range(0,10)
        days = [y*365 for y in years]
        plt.yticks(days,years)
        plt.ylabel('regi_years')
    plt.show()


def getPlots(dataMat, labels, num):
    n, m = np.shape(dataMat)
    Xs = []
    Ys = []
    Zs = []
    for i in range(0, num):
        Xs.append([])
        Ys.append([])
        Zs.append([])
    for (i, label) in enumerate(labels):
        index = int(label)
        Xs[index].append(dataMat[i, 0])
        Ys[index].append(dataMat[i, 1])
        if m > 2:
            Zs[index].append(dataMat[i, 2])
    return Xs, Ys, Zs


def kmean(dataMat, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
          verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto'):
    kmeans = KMeans(n_clusters, init, n_init, max_iter, tol, precompute_distances, verbose, random_state, copy_x,
                    n_jobs, algorithm)
    kmeans.fit(dataMat)
    index = ""
    centers = kmeans.cluster_centers_
    print "聚类中心:\n%s" % centers

    return kmeans.labels_, centers


def getLabelInfoDF(labels):
    ser = pd.value_counts(labels)

    l_df = pd.Series.to_frame(ser)
    l_df.columns = ['个数']
    per = ser.apply(lambda x: "%.2f%%" % (x / float(len(labels)) * 100))
    l_df["占比"] = per
    return l_df


def getCenterDF(centers, lables):
    cent_df = pd.DataFrame(centers, columns=lables)
    return cent_df


def kmeanPlot(dataMat, n_clusters, xlable, ylabe):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(dataMat)

    labels = kmeans.labels_
    l_df = getLabelInfoDF(labels)

    centers = kmeans.cluster_centers_
    c_df = getCenterDF(centers, [xlable, ylabe])

    res = pd.merge(c_df, l_df, left_index=True, right_index=True)
    print res

    plt.scatter(dataMat[:, 0], dataMat[:, 1],
                c=labels.astype(np.float), edgecolor='k', s=20)
    plt.scatter(centers[:, 0], centers[:, 1], c="r", marker="x", edgecolor='k', s=100)

    plt.show


def kmeanAndPlotOrig(dataMat, ranges, minVals, n_clusters, xlable, ylabe, ):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(dataMat)

    labels = kmeans.labels_
    l_df = getLabelInfoDF(labels)

    centers = kmeans.cluster_centers_
    origCenters = transOrig(centers, ranges, minVals)
    c_df = getCenterDF(origCenters, [xlable, ylabe])

    res = pd.merge(c_df, l_df, left_index=True, right_index=True)
    print res

    origDataMat = transOrig(dataMat, ranges, minVals)
    plt.scatter(origDataMat[:, 0], origDataMat[:, 1],
                c=labels.astype(np.float), edgecolor='k', s=20)

    plt.scatter(origCenters[:, 0], origCenters[:, 1], c="r", marker="x", edgecolor='k', s=100)

    plt.show


def getMergeDF(kmeans,ranges, minVals,xlable, ylabe):
    labels = kmeans.labels_
    l_df = getLabelInfoDF(labels)
    centers = kmeans.cluster_centers_
    origCenters = transOrig(centers, ranges, minVals)
    c_df = getCenterDF(origCenters, [xlable, ylabe])

    res = pd.merge(c_df, l_df, left_index=True, right_index=True)
    return res

def kmeanOrigND(dataMat,  n_clusters, vLabels,func,*args ):
    kmeans = KMeans(n_clusters=n_clusters)

    kmeans.fit(dataMat)


    #     userList = list(dataDF_filter2['uid'])
    #     labels = kmeans.labels_
    #     user_ser = pd.Series(userList)
    #     lb_ser=pd.Series(labels)
    #     user_lable_df = pd.DataFrame({"uid":user_ser,"label":lb_ser})
    #     TIMEFORMAT = '%m%d%Y%H%M%s'
    #     t = time.strftime(TIMEFORMAT, time.localtime())
    #     for i in range(0,n_clusters):
    #         c_user = user_lable_df[user_lable_df.label==i]
    #         file_name = "%s_%d_%d.csv"%(t,i,len(c_user))
    #         c_user.uid.to_csv('result/%s'%file_name, sep=',', header=True, index=False)

    labels = kmeans.labels_
    l_df = getLabelInfoDF(labels)

    centers = kmeans.cluster_centers_
    origCenters = func(centers, *args)
    c_df = getCenterDF(origCenters, vLabels)

    res = pd.merge(c_df, l_df, left_index=True, right_index=True)
    print res
    if len(vLabels) == 2:
        OrigdataMat = func(dataMat, *args)
        n_clusters = kmeans.n_clusters
        Xs, Ys, Zs = getPlots(OrigdataMat, labels, n_clusters)
        plotKmean2D(Xs,Ys,lable_dic[vLabels[0]],lable_dic[vLabels[1]],origCenters,n_clusters)
    if len(vLabels) == 3:
        OrigdataMat = func(dataMat, *args)
        n_clusters = kmeans.n_clusters
        Xs, Ys, Zs = getPlots(OrigdataMat, labels, n_clusters)
        plot3D(Xs,Ys,Zs,lable_dic[vLabels[0]],lable_dic[vLabels[1]],lable_dic[vLabels[2]],origCenters,n_clusters)
    return res


def getKMeanResult(dataDF,vLabels):
    dataToKmean = dataDF[vLabels]
    dataMat = np.array(dataToKmean)
    dataMat_scale, scale, mean = scaleNorm(dataMat)
    n_cluster=2*len(vLabels)+1
    kmeans =  kmeanOrigND(dataMat_scale,  n_cluster, vLabels,scaleToOrig,scale, mean )

    return kmeans

def plotBar(barDF):
    plt.figure()
    x = range(0,len(barDF))
    y = barDF["个数"]
    plt.bar(x,y,width=0.8,facecolor="#9999ff",edgecolor="white")
    for x,y in zip(x,y):
        plt.text(x,y+0.05,'%d' % y,ha='center',va='bottom')

    #     colu = list(resDF.columns)
    #     t_l = colu[0:len(colu)-2]
    #     title =  "+".join(t_l) + " 聚类"

    plt.show

if __name__ == '__main__':
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn import preprocessing
    import pandas as pd
    import matplotlib.pyplot as plt
    import time

    #由于matplotlib汉化不是很好,需要对汉子的标题进行转化,需要提供一个转化的字典
    lable_dic = {'阅读': 'read', '原创': 'publish',
                 '转发': 'transmit', '评论': 'comment', '赞': 'like', '关注数': 'attend', '粉丝数': 'fans', '互粉数': 'recip',
                 '注册天数': 'regi_days', '互动数': 'TCL', '主动行为数': 'PTCL'}

    dataDF = pd.read_csv("data/data_source_no_v_low.csv")

    del dataDF["Unnamed: 0"]




    # uid       阅读    原创    转发    评论      赞    关注数    粉丝数   互粉数  注册天数 互动数  主动行为数
    vLabels =["阅读","主动行为数"]
    resDF = getKMeanResult(dataDF_filter2,vLabels)