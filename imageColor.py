#!/usr/bin/env python

import cv2
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from joblib import Parallel, delayed
import multiprocessing
import collections
from tqdm import tqdm
import PIL.Image, PIL.ImageTk
from Tkinter import *
import Tkinter, Tkconstants, tkFileDialog, tkMessageBox

# Hard-coded DPI for my monitor
MY_DPI = 94

# Computation

class FrameExtractor:
    totalFrames = None
    capture = None
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
        print("Frame width : {} px".format(self.FRAME_WIDTH))

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

    def displayChart(self):
        # Turn interactive plotting off
        plt.ioff()
        plt.figure()
        plt.axis("off")
        plt.imshow(self.CHART,aspect='auto')
        fig = plt.gcf()
        return fig


class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None

    def __init__(self, clusters=3):
        self.CLUSTERS = clusters

    def dominantColors(self, imageName, imageIndex):
        self.IMAGE = "images/" + imageName.split(".")[0] + str(imageIndex) + ".jpg"
        print(self.IMAGE)
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

        #print("frame {}/{}".format(self.CHART_INDEX+1, self.TOTAL_FRAMES))

        #creating color rectangles
        #start = 0
        colorTable = []
        for i in range(self.CLUSTERS):
            singleTable = np.zeros((4))
            singleTable[0] = colors[i][0]
            singleTable[1] = colors[i][1]
            singleTable[2] = colors[i][2]
            singleTable[3] = hist[i]
            # Getting rgb values
            colorTable.append(singleTable)

            #print(colorTable)

            # Add colors to chart

            #using cv2.rectangle to plot colors
            #cv2.rectangle(self.CHART, (self.CHART_INDEX * self.FRAME_WIDTH, int(start)), (self.FRAME_WIDTH * (self.CHART_INDEX + 1), int(end)), (r,g,b), -1)
            #start = end

        return colorTable

# GUI
class Window(Frame):
    videoFilename = None
    filename = None
    clusters = None
    frames = None
    isVideoSelected = None
    defaultFrames = None
    defaultClusters = None
    plotName = None
    plot = None
    load = None
    defaultImageWidth = None
    defaultImageHeight = None

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()
        self.isVideoSelected = FALSE
        self.defaultFrames = 10
        self.defaultClusters = 5
        self.defaultImageWidth = 500
        self.defaultImageHeight = 200

    def setFilename(self, name):
        self.videoFilename = name

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget
        self.master.title("GUI")

        # Overriding cross behavior
        root.protocol('WM_DELETE_WINDOW', self.client_exit)

        # creating a menu instance
        menuBar = Menu(self.master)
        self.master.config(menu=menuBar)

        # create the file object)
        file = Menu(menuBar)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Open", command=self.open_file)
        file.add_command(label="Save as", command=self.save_as)
        file.add_command(label="Exit", command=self.client_exit)


        #added "file" to our menu
        menuBar.add_cascade(label="File", menu=file)

        # create the file object)
        edit = Menu(menuBar)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        edit.add_command(label="Show Img", command=self.showImg)
        edit.add_command(label="Show Text", command=self.showText)

        # added "file" to our menu
        menuBar.add_cascade(label="Edit", menu=edit)

        # grid config
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        # add parameter entry fields
        Label(self.master, text="Frames").grid(row=0)
        Label(self.master, text="Clusters").grid(row=1)
        Label(self.master, text="Image width (px)").grid(row=2)
        Label(self.master, text="Image height (px)").grid(row=3)

        self.frames = StringVar()
        self.clusters = StringVar()
        self.imageWidth = StringVar()
        self.imageHeight = StringVar()

        frames = Entry(self.master, textvariable=self.frames)
        clusters = Entry(self.master, textvariable=self.clusters)
        imageWidth = Entry(self.master, textvariable=self.imageWidth)
        imageHeight = Entry(self.master, textvariable=self.imageHeight)

        frames.grid(row=0, column=1)
        clusters.grid(row=1, column=1)
        imageWidth.grid(row=2, column=1)
        imageHeight.grid(row=3, column=1)

        # creating a button instance and placing it
        Button(self.master, text="Compute!",command=self.video_computation).grid(row=4, columnspan=2)


    def showImg(self):
        self.load = PIL.Image.open(self.plotName)
        render = PIL.ImageTk.PhotoImage(self.load)

        # labels can be text or images
        img = Label(self.master, image=render)
        img.image = render
        img.grid(row=5, columnspan=2)


    def showText(self):
        Label(self.master, text="Video path : {}".format(self.videoFilename)).grid(row=5, column=0)

    def client_exit(self):
        root.quit()
        exit()

    def open_file(self):
        videoFilePath = tkFileDialog.askopenfilename(initialdir = "/home/uidq6974/Dev/Python/openCV/imageColor",title = "Select video file",filetypes = (("avi files","*.avi"),("all files","*.*")))
        self.videoFilename = os.path.basename(videoFilePath)
        self.isVideoSelected = TRUE

    def save_as(self):


        if(self.load != None):
            newImage = tkFileDialog.asksaveasfile(filetypes=(("Portable Network Graphics (*.png)", "*.png"),
                                        ("All Files (*.*)", "*.*")), mode='wb', defaultextension=".png")
            extension = self.plotName.rsplit('.', 1)[-1]
            if newImage is None: # asksaveasfile return `None` if dialog closed with "cancel".
                return
            self.load.save(newImage, extension)
            newImage.close()
            print("Image successfully saved to : {}".format(newImage.name))
        else:
            tkMessageBox.showinfo("Info", "Image {} has not been generated yet!".format(self.plotName))
            return



    def video_computation(self):
        if(self.isVideoSelected):
            if not self.frames.get():
                tkMessageBox.showinfo("Info", "No frames number set : using default ({})".format(self.defaultFrames))
                frames = int(self.defaultFrames)
            else :
                frames = int(self.frames.get())
            if(frames < 2):
                tkMessageBox.showinfo("Info", "The number of frames must be more than one. \nUsing defaut ({})".format(self.defaultFrames))
                frames = self.defaultFrames

            if not self.clusters.get():
                tkMessageBox.showinfo("Info", "No clusters number set : using default ({})".format(self.defaultClusters))
                clusters = int(self.defaultClusters)
            else:
                clusters = int(self.clusters.get())

            if not self.imageWidth.get():
                tkMessageBox.showinfo("Info", "No image width set : using default ({})".format(self.defaultImageWidth))
                width = int(self.defaultImageWidth)
            else:
                width = int(self.imageWidth.get())

            if not self.imageHeight.get():
                tkMessageBox.showinfo("Info", "No image height set : using default ({})".format(self.defaultImageHeight))
                height = int(self.defaultImageHeight)
            else:
                height = int(self.imageHeight.get())

            if not os.path.exists("images"):
                os.makedirs("images")

            print(self.videoFilename)
            fe = FrameExtractor(self.videoFilename)
            fe.read()
            maxFrame = fe.getFrames(frames)

            if maxFrame > 0:
                dc = DominantColors(clusters)
                #for i in range(maxFrame):
                    # Retrieve piture name
                    #frameName = "images/" + fileName.split(".")[0] + str(i) + ".jpg"
                num_cores = multiprocessing.cpu_count()
                colors = Parallel(n_jobs=num_cores-2)(delayed(dc.dominantColors)(self.videoFilename, i) for i in range(maxFrame))
                self.save_picture(colors, maxFrame, clusters, width, height)

            fe.purge()
            self.showImg()
        else:
            tkMessageBox.showinfo("Info", "Please select a video first (file > open)")
            self.open_file()

    def save_picture(self, colors, maxFrame, clusters, width, height, path=""):

        fullChart = Chart(maxFrame,clusters, width, height)
        fullChart.addFrames(colors)
        self.plot = fullChart.displayChart()
        self.plotName = self.videoFilename.split(".")[0] + ".png"
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.plot.savefig(self.plotName, transparent = True)




root = Tk()
#size of the window
root.geometry("1000x700")

app = Window(root)
#app.setFilename(videoFilename)
root.mainloop()
