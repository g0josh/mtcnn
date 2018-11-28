from PIL import ImageDraw
import cv2

def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """
        Draw bounding boxes and facial landmarks.
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    for b in bounding_boxes:
        draw.rectangle([(b[0], b[1]), (b[2], b[3])],
            outline='red')

    for p in facial_landmarks:
        for i in range(5):
            draw.ellipse([(p[i] - 1.0, p[i + 5] - 1.0),
                          (p[i] + 1.0, p[i + 5] + 1.0)],
                          outline='blue')
    return img_copy

def _show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """
        Draw bounding boxes and facial landmarks.
    """
    img_copy = img.copy()
    for b in bounding_boxes:
        b = [int(x) for x in b]
        cv2.rectangle(img_copy, (b[0], b[1]), (b[2], b[3]), (0,0,255), 2)

    for p in facial_landmarks:
        for i in range(5):
            cv2.circle(img_copy, (p[i], p[i+5]), 3, (0,255,0), 1)
    return img_copy
