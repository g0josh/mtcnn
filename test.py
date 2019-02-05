from __future__ import print_function
from __future__ import division

from detector import FaceDetector
from utils.viz_utils import drawBoxesGetFaces, writeText
import cv2
import time
import torch

def main():
    face_detector = FaceDetector(device='cuda')
    cap = cv2.VideoCapture(-1)
    fr_cnt = 0
    st_time = time.time()
    interval = 5
    info = 'calculating...'
    faces = torch.zeros((112,112))
    cnt = 0
    totime = time.time()
    while cap.isOpened() and time.time()-totime < 60:
        isSuccess, image = cap.read()
        if isSuccess:
            fr_cnt += 1
            cnt += 1
            bounding_boxes,landmarks = face_detector.detect(image)
            if bounding_boxes is not None and bounding_boxes.numel() > 0:
                image, faces = drawBoxesGetFaces(image, bounding_boxes, landmarks)

        if time.time() - st_time >= interval:
            info = str(fr_cnt/interval) + "FPS | " + str(image.shape)
            print (info)
            fr_cnt = 0
            st_time = time.time()

        image = writeText(image, info)
        cv2.imshow('debug', image)
        cv2.imshow('face', faces[0])
        if cv2.waitKey(1)&0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
