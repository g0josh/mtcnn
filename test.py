from __future__ import print_function
from src.detector import detect_faces
from utils.visualization_utils import _show_bboxes
import cv2

def main():
    #image = Image.open('images/test2.jpg')
    cap = cv2.VideoCapture(0)
    # cap.set(3,1280)
    # cap.set(4,720)
    while cap.isOpened():
        isSuccess, image = cap.read()
        if isSuccess:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bounding_boxes, landmarks = detect_faces(image)
            # print("bbox = {}\n{}\nland={}\n{}".format(type(bounding_boxes), bounding_boxes, type(landmarks), landmarks))
            if bounding_boxes.numel() > 0:
                image = _show_bboxes(image, bounding_boxes, landmarks)
            #image.show()
            cv2.imshow('face cap', image)
            if cv2.waitKey(1)&0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()
