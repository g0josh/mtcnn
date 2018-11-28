from __future__ import print_function
from src.detector import detect_faces
from utils.visualization_utils import _show_bboxes
from PIL import Image
import cv2
import numpy

def main():
    #image = Image.open('images/test2.jpg')
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            # print ("frame type = {}, size = {}".format(type(frame), frame.size))
            # image = Image.fromarray(frame)
            # print ("type = {}, size-{}".format(type(image), image.size))
            # exit()
            bounding_boxes, landmarks = detect_faces(frame)
            print("bbox = {}\n{}\nland={}\n{}".format(type(bounding_boxes), bounding_boxes, type(landmarks), landmarks))
            image = _show_bboxes(frame, bounding_boxes, landmarks)
            #image.show()
            cv2.imshow('face cap', image)
            if cv2.waitKey(1)&0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()
