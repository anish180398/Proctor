import cv2
import numpy as np
import dlib
from math import hypot
from mouth_tracking import *
from facial_landmarks_detection import *
from blink_detection import *
from gaze_detection import *
from headposeDetection import *
#from object_detection import *
print("Started")
cap = cv2.VideoCapture(0)

if(cap.isOpened()==False):
    cap.open()

while True:

	ret, frame = cap.read()
	
	faceCount, faces = detectFace(frame)
	mouthTrack(faces, frame)
	headPoseDetect(ret, frame)
	blinkStatus = isBlinking(faces, frame)
	eyeStatus = gazeDetection(faces, frame)
	headPose = headPoseDetect(ret, frame)
	#objectStatus = detectObject(frame)
	#print(blinkStatus[2]+' - '+eyeStatus)
	#print(objectStatus)
	print(eyeStatus)
	print(headPose)



	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()