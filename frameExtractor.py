#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Hard-coded DPI for my monitor
MY_DPI = 94

class FrameExtractor:
    totalFrames = None
    capture = None
    def __init__(self, fileName):
        self.fileName = fileName

    def read(self):

        cap = cv2.VideoCapture(self.fileName)
        logging.info("video : {}".format(self.fileName))
        if not cap.isOpened():
            logging.error('Could not open {}'.format(self.fileName))
            return
        self.capture = cap
        self.totalFrames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info("{} FRAMES".format(self.totalFrames))

    def getFrames(self, divisions):
        # Will extract divisions equally reparted frames from video (format : jpg)
        framePeriod = self.totalFrames / divisions
        logging.info("Select one frame every {}".format(int(framePeriod)))
        # Trial
        success,image = self.capture.read()
        count = 0
        while success:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(count * framePeriod))
            success,image = self.capture.read()
            if success == True:
                logging.info("Frame number {}".format(int(count * framePeriod)))
                frameName = "images/" + os.path.basename(self.fileName).split(".")[0] + "_" + str(count) + ".jpg"
                cv2.imwrite(frameName, image)     # save frame as JPEG file
                count += 1

        return count

    def getPlot(self, maxFrame, clusters, width, height, colors):
        fullChart = Chart(maxFrame,clusters, width, height)
        fullChart.addFrames(colors)
        plot = fullChart.createChart()
        return plot

    def purge(self):
        dirPath = "images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath+"/"+fileName)

class Chart:
    CHART = None
    TOTAL_FRAMES = None
    FRAME_WIDTH = None
    TOTAL_WIDTH = None
    CHART_INDEX = None
    RGB_MAP = None
    CLUSTER = None
    IMAGE_WIDTH = None
    IMAGE_HEIGHT = None

    # Set global chart size
    def __init__(self, totalFrames, cluster, width, height):
        self.TOTAL_FRAMES = totalFrames
        self.IMAGE_WIDTH = width
        self.IMAGE_HEIGHT = height

        if self.TOTAL_FRAMES > self.IMAGE_WIDTH:
            # if too much frames for the picture, 1 frame = 1px
            self.FRAME_WIDTH = 1
            self.TOTAL_WIDTH = self.TOTAL_FRAMES
        else:
            # calculate frame color representation width
            self.FRAME_WIDTH = int((self.IMAGE_WIDTH / totalFrames) + 0.5)
            # compute again chart width to adjust to previous rounding
            self.TOTAL_WIDTH = self.FRAME_WIDTH * self.TOTAL_FRAMES

        # Create empty chart
        self.CHART = np.zeros((self.IMAGE_HEIGHT, self.TOTAL_WIDTH, 3), np.uint8)
        self.CHART_INDEX = 0

        self.CLUSTER = cluster

        # Store all data for the map
        self.RGB_MAP = np.zeros((self.TOTAL_FRAMES, self.CLUSTER, 4), np.uint8)

    def addFrames(self, fullColorTable):
        self.RGB_MAP = fullColorTable
        for frame in self.RGB_MAP:
            start = 0
            end = 0

            for i in range(self.CLUSTER):
                end = start + frame[i][3] * self.IMAGE_HEIGHT
                cv2.rectangle(self.CHART, (self.CHART_INDEX * self.FRAME_WIDTH, int(start)), (self.FRAME_WIDTH * (self.CHART_INDEX + 1), int(end)), (frame[i][0],frame[i][1],frame[i][2]), -1)
                start = end
            self.CHART_INDEX += 1

    def createChart(self):
        # Turn interactive plotting off
        plt.ioff()
        plt.figure(figsize=(self.IMAGE_WIDTH/MY_DPI, self.IMAGE_HEIGHT/MY_DPI), dpi=MY_DPI)
        plt.axis("off")
        plt.imshow(self.CHART,aspect='auto')
        fig = plt.gcf()
        return fig
