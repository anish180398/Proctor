
from argparse import ArgumentParser

import cv2

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator

def headPose(cap):

    # cap = cv2.VideoCapture(video_src)

    # Get the frame size. This will be used by the pose estimator.
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 2. Introduce a pose estimator to solve pose.
    pose_estimator = PoseEstimator(img_size=(height, width))

    # 3. Introduce a mark detector to detect landmarks.
    mark_detector = MarkDetector()

    # 4. Measure the performance with a tick meter.
    tm = cv2.TickMeter()

    # Now, let the frames flow.
    # while True:

    #     # Read a frame.
    frame_got, frame = cap.read()
    # if frame_got is False:
    #     break

    # # If the frame comes from webcam, flip it so it looks like a mirror.
    # if video_src == 0:
    #     frame = cv2.flip(frame, 2)

        # Step 1: Get a face from current frame.
    facebox = mark_detector.extract_cnn_facebox(frame)

    # Any face found?
    if facebox is not None:

        # Step 2: Detect landmarks. Crop and feed the face area into the
        # mark detector.
        x1, y1, x2, y2 = facebox
        face_img = frame[y1: y2, x1: x2]

        # Run the detection.
        tm.start()
        marks = mark_detector.detect_marks(face_img)
        tm.stop()

        # Convert the locations from local face area to the global image.
        marks *= (x2 - x1)
        marks[:, 0] += x1
        marks[:, 1] += y1

        # Try pose estimation with 68 points.
        pose = pose_estimator.solve_pose_by_68_points(marks)

        # All done. The best way to show the result would be drawing the
        # pose on the frame in realtime.

        # Do you want to see the pose annotation?
        pose_estimator.draw_annotation_box(
            frame, pose[0], pose[1], color=(0, 255, 0))

        # Do you want to see the head axes?
        # pose_estimator.draw_axes(frame, pose[0], pose[1])

        # Do you want to see the marks?
        mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

        # Do you want to see the facebox?
        mark_detector.draw_box(frame, [facebox])
    return pose[0]
    # Show preview.
    # cv2.imshow("Preview", frame)
    # if cv2.waitKey(1) == 27:
    #     break
