from __future__ import print_function
from src.detector import FaceDetector
from utils.viz_utils import show_bboxes
import cv2

def main():
    #image = Image.open('images/test2.jpg')
    face_detector = FaceDetector(device='cuda')
    cap = cv2.VideoCapture(0)
    # cap.set(3,1280)
    # cap.set(4,720)
    while cap.isOpened():
        isSuccess, image = cap.read()
        if isSuccess:
            bounding_boxes,landmarks = face_detector.detect(image)
            if bounding_boxes.numel() > 0:
                image = show_bboxes(image, bounding_boxes, landmarks)
            cv2.imshow('face cap', image)
            if cv2.waitKey(1)&0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()
