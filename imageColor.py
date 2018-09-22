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
import tkinter
import tkinter.messagebox
import tkinter.filedialog
import logging
import faceDetection

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

# Face detection module : frames detected faces in video output
class FaceDetection:
    FILENAME = None
    IMWIDTH = None
    IMHEIGHT = None
    FPS = None
    TOTALFRAMES = None
    CAPTURE = None
    OUTPUT = None

    def set_input(self):
        # Read through input video
        cap = cv2.VideoCapture(self.FILENAME)
        if not cap.isOpened():
            logging.error('Could not open {}'.format(self.FILENAME))
            return
        self.CAPTURE = cap
        # read initial frame parameters
        self.IMWIDTH = int(self.CAPTURE.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.IMHEIGHT = int(self.CAPTURE.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.FPS = self.CAPTURE.get(cv2.CAP_PROP_FPS)
        self.TOTALFRAMES = int(self.CAPTURE.get(cv2.CAP_PROP_FRAME_COUNT))

    def set_output(self, output = "Output", extension = "avi", fourccEncoding = "XVID"):
          # set up output file
        fourcc = cv2.VideoWriter_fourcc(*fourccEncoding)
        outputName = "output/videos/" + output + "." + extension
        logging.info("Output file : {} ({})".format(outputName, fourccEncoding))
        self.OUTPUT = cv2.VideoWriter(outputName,fourcc, self.FPS, (self.IMWIDTH, self.IMHEIGHT))

    def compute(self, filename):
        # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
        # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
        face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        self.FILENAME = filename
        self.set_input()
        self.set_output("test", "avi", "XVID")
        frame_number = -1
        success = True

        while success:
            # read first frame
            success,frame = self.CAPTURE.read()
            frame_number += 1
            if success == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                self.OUTPUT.write(frame)
                if frame_number % 100 == 0:
                    logging.info("frame {}/{}".format(frame_number, self.TOTALFRAMES))
        logging.info("Done !")
        self.CAPTURE.release()
        self.OUTPUT.release()

# GUI
class Window(tkinter.Frame):
    videoFilename = None
    videoFilePath = None
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
    frameExtractor = None

    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        self.master = master
        self.init_window()
        self.isVideoSelected = False
        self.defaultFrames = 10
        self.defaultClusters = 5
        self.defaultImageWidth = 500
        self.defaultImageHeight = 200

    def setFilename(self, name):
        self.videoFilename = name

    #Creation of init_window
    def init_window(self):

        # changing the title of our master widget
        self.master.title("Color Movie")

        # Overriding cross behavior
        root.protocol('WM_DELETE_WINDOW', self.client_exit)

        # creating a menu instance
        menuBar = tkinter.Menu(self.master)
        self.master.config(menu=menuBar)

        # create the file object)
        file = tkinter.Menu(menuBar)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Open", command=self.open_file)
        file.add_command(label="Save as", command=self.save_as)
        file.add_command(label="Detect faces", command=self.detect_faces)
        file.add_command(label="Extract faces", command=self.extract_faces)
        file.add_command(label="Exit", command=self.client_exit)

 
        #added "file" to our menu
        menuBar.add_cascade(label="File", menu=file)

        # create the file object)
        edit = tkinter.Menu(menuBar)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        edit.add_command(label="Show Img", command=self.showImg)
        edit.add_command(label="Show Text", command=self.showText)

        # added "file" to our menu
        menuBar.add_cascade(label="Edit", menu=edit)

        # grid config
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=3)
        root.grid_columnconfigure(2, weight=1)

        # add parameter entry fields
        tkinter.Label(self.master, text="Frames").grid(row=0)
        tkinter.Label(self.master, text="Clusters").grid(row=1)
        tkinter.Label(self.master, text="Image width (px)").grid(row=2)
        tkinter.Label(self.master, text="Image height (px)").grid(row=3)
        tkinter.Label(self.master, text="Video file").grid(row=4)

        self.frames = tkinter.StringVar()
        self.clusters = tkinter.StringVar()
        self.imageWidth = tkinter.StringVar()
        self.imageHeight = tkinter.StringVar()
        self.videoFilePath = tkinter.StringVar()

        frames = tkinter.Entry(self.master, textvariable=self.frames)
        clusters = tkinter.Entry(self.master, textvariable=self.clusters)
        imageWidth = tkinter.Entry(self.master, textvariable=self.imageWidth)
        imageHeight = tkinter.Entry(self.master, textvariable=self.imageHeight)
        filePath = tkinter.Entry(self.master, textvariable=self.videoFilePath)

        frames.grid(row=0, column=1, sticky='we')
        clusters.grid(row=1, column=1, sticky='we')
        imageWidth.grid(row=2, column=1, sticky='we')
        imageHeight.grid(row=3, column=1, sticky='we')
        filePath.grid(row=4, column=1, sticky='we')

        # add buttons
        tkinter.Button(self.master, text="Select file",command=self.open_file).grid(row=4, column=2)
        tkinter.Button(self.master, text="Compute!",command=self.video_computation).grid(row=5, columnspan=3)


    def showImg(self):
        self.load = PIL.Image.open("output/shades/" + self.plotName)
        render = PIL.ImageTk.PhotoImage(self.load)

        # labels can be text or images
        img = tkinter.Label(self.master, image=render)
        img.image = render
        img.grid(row=6, columnspan=3)


    def showText(self):
        tkinter.Label(self.master, text="Video path : {}".format(self.videoFilename)).grid(row=6, column=0)

    def client_exit(self):
        # Delete all pictures in images folder
        if (self.frameExtractor != None):
            self.frameExtractor.purge()
        root.quit()

    def open_file(self):
        self.videoFilePath.set(tkinter.filedialog.askopenfilename(initialdir = "/home/uidq6974/Dev/Python/openCV/imageColor",title = "Select video file",filetypes = (("avi files","*.avi"),("all files","*.*"))))
        self.videoFilename = os.path.realpath(str(self.videoFilePath.get()))
        # Instanciate frame extractor with video name
        self.frameExtractor = FrameExtractor(self.videoFilename)
        self.isVideoSelected = True


    def save_as(self):


        if(self.load != None):
            newImage = tkinter.filedialog.asksaveasfile(filetypes=(("Portable Network Graphics (*.png)", "*.png"),
                                        ("All Files (*.*)", "*.*")), mode='wb', defaultextension=".png")
            extension = self.plotName.rsplit('.', 1)[-1]
            if newImage is None: # asksaveasfile return `None` if dialog closed with "cancel".
                return
            self.load.save(newImage, extension)
            newImage.close()
            logging.info("Image saved to : {}".format(newImage.name))
        else:
            tkinter.messagebox.showinfo("Info", "Image {} has not been generated yet!".format(self.plotName))
            return



    def video_computation(self):
        if(self.isVideoSelected):
            if not self.frames.get():
                tkinter.messagebox.showinfo("Info", "No frames number set : using default ({})".format(self.defaultFrames))
                frames = int(self.defaultFrames)
            else :
                frames = int(self.frames.get())
            if(frames < 2):
                tkinter.messagebox.showinfo("Info", "The number of frames must be more than one. \nUsing defaut ({})".format(self.defaultFrames))
                frames = self.defaultFrames

            if not self.clusters.get():
                tkinter.messagebox.showinfo("Info", "No clusters number set : using default ({})".format(self.defaultClusters))
                clusters = int(self.defaultClusters)
            else:
                clusters = int(self.clusters.get())

            if not self.imageWidth.get():
                tkinter.messagebox.showinfo("Info", "No image width set : using default ({})".format(self.defaultImageWidth))
                width = int(self.defaultImageWidth)
            else:
                width = int(self.imageWidth.get())

            if not self.imageHeight.get():
                tkinter.messagebox.showinfo("Info", "No image height set : using default ({})".format(self.defaultImageHeight))
                height = int(self.defaultImageHeight)
            else:
                height = int(self.imageHeight.get())

            if not os.path.exists("images"):
                os.makedirs("images")

            # create output folder
            if not os.path.exists("output/shades"):
                os.makedirs("output/shades")

            logging.info(self.videoFilename)
            self.frameExtractor.read()
            maxFrame = self.frameExtractor.getFrames(frames)

            if maxFrame > 0:
                dc = DominantColors(clusters)
                num_cores = multiprocessing.cpu_count()
                colors = []
                num_cores = multiprocessing.cpu_count()
                colors = Parallel(n_jobs=num_cores-1)(delayed(dc.dominantColors)
                                                      (self.videoFilename, i) for i in tqdm(range(maxFrame)))

                self.save_picture(colors, maxFrame, clusters, width, height)

            self.showImg()
        else:
            tkinter.messagebox.showinfo("Info", "Please select a video first (file > open)")
            self.open_file()

    def save_picture(self, colors, maxFrame, clusters, width, height, path=""):

        fullChart = Chart(maxFrame,clusters, width, height)
        fullChart.addFrames(colors)
        self.plot = fullChart.createChart()
        self.plotName = os.path.basename(self.videoFilename).split(".")[0]+"_shade.png"
        logging.info("Output file : {}".format("output/shades/" + self.plotName))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.plot.savefig("output/shades/" + self.plotName, transparent = True)

    def extract_faces(self):
        # create output folder
        if not os.path.exists("output/faces"):
            os.makedirs("output/faces")

        # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
        # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
        face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')

        folderName = "./images/"
        for file in sorted(os.listdir(folderName)):
            if file.endswith(".jpg"):
                img = cv2.imread(folderName+file, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                i = 0
                for (x, y, w, h) in faces:
                    logging.info(file + "\t contains face(s)")
                    roi_color = img[y:y+h, x:x+w]
                    # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    cv2.imwrite('output/faces/face_' + file + "_" + str(i)+".png", roi_color)
                    i += 1
                # cv2.imshow('img',img)
                cv2.destroyAllWindows()
    
    def detect_faces(self):
        # create output folder
        if not os.path.exists("output/videos"):
                os.makedirs("output/videos")
# initialize log
logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s',datefmt='%d/%m/%Y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

# initialize GUI

root = tkinter.Tk()
# size of the window
root.geometry("1000x700")
# GUI icon
root.iconbitmap(r'MovieColor.ico')
app = Window(root)
root.mainloop()
        detection = faceDetection.FaceDetection()
        detection.compute(self.videoFilename)   

