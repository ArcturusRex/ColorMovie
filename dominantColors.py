#!/usr/bin/env python

import cv2
from sklearn.cluster import KMeans
import numpy as np
import os
import collections
from tqdm import tqdm
import logging

# Computation of dominant colors
class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, clusters=3):
        self.CLUSTERS = clusters

    def dominantColors(self, imageName, imageIndex):
        self.IMAGE = "images/" + os.path.basename(imageName).split(".")[0] + "_" + str(imageIndex) + ".jpg"
        #read image
        img = cv2.imread(self.IMAGE)
        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #reshaping to a list of pixels
        img = img.reshape(img.shape[0] * img.shape[1],3)

        #save image after operations
        self.IMAGE = img
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS, max_iter = 10)
        kmeans.fit(img)
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        #save labels
        self.LABELS = kmeans.labels_
        colorTable = self.setHistogram()
        return colorTable

    def setHistogram(self):

        #labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)
        #create frequency count tables
        (hist,bounds) = np.histogram(self.LABELS, bins = numLabels, density=True)
        #appending frequencies to cluster centers
        colors = self.COLORS

        indexes = range(len(colors))
        colorDict = dict(zip(indexes, colors))
        #descending order sorting as per frequency count
        finalDict = collections.OrderedDict(sorted(colorDict.items(),key=lambda i:sum(i[1]),reverse=False))
        colors = list(finalDict.values())

        colors = sorted(colors, key=lambda colors: colors[0]+colors[1]+colors[2])
        hist = [ hist[i] for i in finalDict.keys()]

        colorTable = []
        for i in range(self.CLUSTERS):
            singleTable = np.zeros((4))
            singleTable[0] = colors[i][0]
            singleTable[1] = colors[i][1]
            singleTable[2] = colors[i][2]
            singleTable[3] = hist[i]
            # Getting rgb values
            colorTable.append(singleTable)
        return colorTable