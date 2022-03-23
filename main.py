import cv2
import numpy as np
import dlib
from math import hypot
from headPose import *
from facial_landmarks_detection import *
from blink_detection import *
from gaze_detection import *
from headposeDetection import *
from phone_detector import *
from detect_open_mouth import *
print("Started")
cap = cv2.VideoCapture(0)

if(cap.isOpened()==False):
    cap.open()

while True:

	ret, frame = cap.read()
	
	faceCount, faces = detectFace(frame)
	#print(faceCount)
	# print(faces)
	# mouthTrack(faces, frame)
	mouthOpen = mouthopen(frame)
	# headPose = headPose(cap)
	# blinkStatus = isBlinking(faces, frame)
	eyeStatus = gazeDetection(faces, frame)
	# headPose = headPoseDetect(ret, frame)
	#objectStatus = detectObject(frame)
	# #print(blinkStatus[2]+' - '+eyeStatus)
	mobDetect = mobDetect(frame)

	if mouthOpen != '' or mouthOpen != 'None':
		print(mouthOpen)
	# print(headPose)
	if eyeStatus != '' or eyeStatus != 'None':
		print(eyeStatus)
	print(mobDetect)


	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()