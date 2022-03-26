from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import numpy as np
import cv2
import requests
from influxdb import InfluxDBClient
from datetime import datetime
import json


'''
>>> from imutils import face_utils
>>> face_utils.FACIAL_LANDMARKS_68_IDXS
OrderedDict([('mouth', (48, 68)), ('inner_mouth', (60, 68)), 
('right_eyebrow', (17, 22)), ('left_eyebrow', (22, 27)), 
('right_eye', (36, 42)), ('left_eye', (42, 48)), 
('nose', (27, 36)), ('jaw', (0, 17))])
'''


face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

thresh = 0.25
mouth_thresh = 0.8
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]
(imStart, imEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["inner_mouth"]


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
def mouth_aspect_ratio(im):
    A = distance.euclidean(im[3], im[5])
    B = distance.euclidean(im[2], im[6])
    C = distance.euclidean(im[1], im[7])
    D = distance.euclidean(im[0], im[4])
    mar = (A + B + C) / (2.0 * D)
    return mar

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.ear_current_frame = 0
        self.counter = 0
        self.flag = 0
        self.mouth_size = 0
        self.IFDBClient = InfluxDBClient('localhost', 8086, 'FaceDetectdb')
        self.IFDBClient.create_database('FaceDetectdb')
        self.IFDBClient.switch_database('FaceDetectdb')
    
    def __del__(self):
        self.video.release()
    


    def get_frame(self):

        success, frame = self.video.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:

            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)#converting to NumPy Array

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            imouth = shape[imStart:imEnd]

            self.mouth_size = mouth_aspect_ratio(imouth)

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            self.ear_current_frame = ear

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            mouthHull = cv2.convexHull(imouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            if self.ear_current_frame < thresh or self.mouth_size > mouth_thresh:
                self.flag += 1
                if self.flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10,325),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    
                    #write_db()
                    jsondata = []
                    data = {
                        "measurement": "driverdata",
                        "tags": {
                            "Name": "John Doe" 
                            },
                        "time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        "fields": {
                            'EAR': self.ear_current_frame
                        }
                    }
                    jsondata.append(data)
                    print ("Writing to DB")
                    self.IFDBClient.write_points(jsondata)

                    r = requests.post('https://maker.ifttt.com/trigger/drowsiness_detection/with/key/bb4Vu9I7TpwnQaVsorvK2u')
                    print ("Sending email alert")


            else:
                self.flag = 0
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def get_MAR(self):
        return self.mouth_size

    def get_EAR(self):
        return self.ear_current_frame


