import cv2
import numpy as np
import time

print('Object Detection import Success!!')
# net has the YOLO loaded
net = cv2.dnn.readNet("weights/yolov3-tiny.weights","config/yolov3-tiny.cfg")

# Classes that'll be detected using object Detection module.
label_classes = []
with open("objectLabels/coco.names","r") as file:
	label_classes = [name.strip() for name in file.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[layer[0]-1] for layer in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0,255,size=(len(label_classes),3))

font = cv2.FONT_HERSHEY_PLAIN
start_time = time.time()
frame_id = 0


def detectObject(frame):

	"""
	Input : frame from the live video stream
	Output : Objects Detected in this frame with their respective Confidences. 
	Datatypes : 
		- Input : Image Frame
		- Output : List of tuples.
	"""

	labels_this_frame = []

	height,width,channels = frame.shape

	blob = cv2.dnn.blobFromImage(frame, 0.00392, (220,220), (0,0,0), True, crop=False)

	# Feeding Blob as an input to our Yolov3-tiny model
	net.setInput(blob)
	# Output labels received at the output of model.
	outs = net.forward(output_layers)


	# show informations on screen
	class_ids = []
	confidences = []
	boxes = []
	for out in outs:
		for detection in out:
			scores = detection[5:]
			class_id = np.argmax(scores)
			confidence = scores[class_id]
			if confidence > 0.5:
				# object detected
				center_x = int(detection[0]*width)
				center_y = int(detection[1]*height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

                # Rectangle coordinates
				x = int(center_x - w / 2)
				y = int(center_y - h / 2)

				boxes.append([x,y,w,h])
				confidences.append(float(confidence))
				class_ids.append(class_id)


	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

	for i in range(len(boxes)):
		if i in indexes:
	        # show the box only if it comes in non-max Suppression Box.
	 		# x, y, w, h = boxes[i]
			label = str(label_classes[class_ids[i]])
	 		# color = colors[class_ids[i]]
			labels_this_frame.append((label,confidences[i]))
	 		# cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
	 		# cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

	return labels_this_frame