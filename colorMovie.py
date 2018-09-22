#!/usr/bin/env python

import cv2
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import PIL.Image, PIL.ImageTk
import tkinter
import tkinter.messagebox
import tkinter.filedialog
import logging
import faceDetection
import frameExtractor
import dominantColors

# GUI
class Window(tkinter.Tk):
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

    def __init__(self, *args, **kwargs):
        tkinter.Tk.__init__(self, *args, **kwargs)
        tkinter.Frame(self)
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
        self.title("Color Movie")

        # Overriding cross behavior
        self.protocol('WM_DELETE_WINDOW', self.client_exit)

        # creating a menu instance
        menuBar = tkinter.Menu(self.master)
        self.config(menu=menuBar)

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
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_columnconfigure(2, weight=1)

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
        self.quit()

    def open_file(self):
        self.videoFilePath.set(tkinter.filedialog.askopenfilename(initialdir = "/home/uidq6974/Dev/Python/openCV/imageColor",title = "Select video file",filetypes = (("avi files","*.avi"),("all files","*.*"))))
        self.videoFilename = os.path.realpath(str(self.videoFilePath.get()))
        # Instanciate frame extractor with video name
        self.frameExtractor = frameExtractor.FrameExtractor(self.videoFilename)
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
                dc = dominantColors.DominantColors(clusters)
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

        plot = self.frameExtractor.getPlot(maxFrame,clusters, width, height, colors)
        self.plotName = os.path.basename(self.videoFilename).split(".")[0]+"_shade.png"
        logging.info("Output file : {}".format("output/shades/" + self.plotName))
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plot.savefig("output/shades/" + self.plotName, transparent = True)

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
        detection = faceDetection.FaceDetection()
        detection.compute(self.videoFilename)   

def main():
    # initialize log
    logging.basicConfig(format='%(asctime)s-%(levelname)s: %(message)s',datefmt='%d/%m/%Y %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)

    app = Window()
    app.geometry("1000x700")
    # GUI icon
    app.iconbitmap(r'MovieColor.ico')
    app.mainloop()

if __name__ == '__main__':
    main()
