#!/usr/bin/env python

import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
import collections

class FrameExtractor:
    def __init__(self, fileName):
        self.fileName = fileName

    def read(self):
        cap = cv2.VideoCapture(self.fileName)
        if not cap.isOpened():
            print('Could not open {}'.format(self.fileName))
            return
        self.capture = cap
        self.totalFrames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("======> {} FRAMES".format(self.totalFrames))

    def getFrames(self, divisions):
        # Will extract divisions equally reparted frames from video (format : jpg)
        framePeriod = self.totalFrames / divisions
        print("Select one frame every {}".format(int(framePeriod)))
        # Trial
        success,image = self.capture.read()
        count = 0
        while success:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(count * framePeriod))
            success,image = self.capture.read()
            if success == True:
                print("Frame number {}".format(count * framePeriod))
                frameName = "images/" + self.fileName.split(".")[0] + str(count) + ".jpg"
                cv2.imwrite(frameName, image)     # save frame as JPEG file
                count += 1
            print('Read a new frame: {}'.format(success))

        return count-1

    def purge(self):
        dirPath = "images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath+"/"+fileName)




class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    CHART = None
    CHART_INDEX = None
    TOTAL_FRAMES = None
    FRAME_WIDTH = None
    TOTAL_WIDTH = None

    def __init__(self, totalFrames, clusters=3):
        self.CLUSTERS = clusters
        self.CHART_INDEX = 0
        self.TOTAL_FRAMES = totalFrames
        if self.TOTAL_FRAMES > 500:
            # if too much frames for the picture, 1 frame = 1px
            self.FRAME_WIDTH = 1
            self.TOTAL_WIDTH = self.TOTAL_FRAMES
        else:
            # calculate frame color representation width
            self.FRAME_WIDTH = int((500 / totalFrames) + 0.5)
            # compute again chart width to adjust to previous rounding
            self.TOTAL_WIDTH = self.FRAME_WIDTH * self.TOTAL_FRAMES
        print("Frame width : {} px".format(self.FRAME_WIDTH))

    def load(self, image):
        self.IMAGE = image

    def dominantColors(self):

        #read image
        img = cv2.imread(self.IMAGE)
        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #reshaping to a list of pixels
        img = img.reshape(img.shape[0] * img.shape[1],3)

        #save image after operations
        self.IMAGE = img

        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)

        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_

        #save labels
        self.LABELS = kmeans.labels_

        #returning after converting to integer from float
        return self.COLORS.astype(int)

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

        #colors = colors[(hist).argsort()]
        #hist = hist[(hist).argsort()]
        hist = [ hist[i] for i in finalDict.keys()]

        #creating empty chart if first frame, modify it otherwise
        if(self.CHART_INDEX == 0):
            self.CHART = np.zeros((200, self.TOTAL_WIDTH, 3), np.uint8)

        #creating color rectangles
        start = 0
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 200

            #getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]

            #using cv2.rectangle to plot colors
            cv2.rectangle(self.CHART, (self.CHART_INDEX * self.FRAME_WIDTH, int(start)), (self.FRAME_WIDTH * (self.CHART_INDEX + 1), int(end)), (r,g,b), -1)
            start = end

        self.CHART_INDEX +=1

    def displayChart(self):
        plt.figure()
        plt.axis("off")
        plt.imshow(self.CHART)
        fig = plt.gcf()
        plt.show()
        return fig

if(len(sys.argv) != 4):
    print("Usage : imageColor.py <filename_with_extension> <number_of_frames> <number_of_color_clusters>")


if not os.path.exists("images"):
    os.makedirs("images")

fileName = sys.argv[1]
fe = FrameExtractor(fileName)
fe.read()
maxFrame = fe.getFrames(int(sys.argv[2]))
clusters = int(sys.argv[3])

if maxFrame > 0:
    dc = DominantColors(maxFrame, clusters)
    for i in tqdm(range(maxFrame)):
        # Retrieve piture name
        frameName = "images/" + fileName.split(".")[0] + str(i) + ".jpg"
        dc.load(frameName)
        colors = dc.dominantColors()
        dc.setHistogram()
    plot = dc.displayChart()
    plotName = fileName.split(".")[0] + ".png"
    plot.savefig(plotName)
fe.purge()