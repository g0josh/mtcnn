from __future__ import print_function
from __future__ import division

from src.detector import FaceDetector
from utils.viz_utils import draw_boxes, write_text
import cv2
import time

def main():
    face_detector = FaceDetector(device='cpu')
    cap = cv2.VideoCapture(0)
    fr_cnt = 0
    st_time = time.time()
    interval = 5
    info = 'calculating...'
    while cap.isOpened():
        isSuccess, image = cap.read()
        if isSuccess:
            fr_cnt += 1
            bounding_boxes,landmarks = face_detector.detect(image)
            if bounding_boxes is not None and bounding_boxes.numel() > 0:
                image = draw_boxes(image, bounding_boxes, landmarks)

        if time.time() - st_time >= interval:
            info = str(fr_cnt/interval) + "FPS | " + str(image.shape)
            # print (info)
            fr_cnt = 0
            st_time = time.time()

        image = write_text(image, info)
        cv2.imshow('face', image)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    cap.release()

if __name__ == "__main__":
    main()
