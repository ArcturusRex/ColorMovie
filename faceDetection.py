import cv2
import logging

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
            logger.error('Could not open {}'.format(self.FILENAME))
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
        logger.info("Output file : {} ({})".format(outputName, fourccEncoding))
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
                    logger.info("frame {}/{}".format(frame_number, self.TOTALFRAMES))
        logger.info("Done !")
        self.CAPTURE.release()
        self.OUTPUT.release()

logger = logging.getLogger(__name__)
