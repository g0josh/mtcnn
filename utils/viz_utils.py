import cv2

def draw_boxes(img, bounding_boxes, facial_landmarks=[]):
    """
        Draw bounding boxes and facial landmarks.
    """
    img_copy = img
    for b in bounding_boxes:
        b = [int(x) for x in b]
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0,0,255), 2)

    for p in facial_landmarks:
        for i in range(5):
            cv2.circle(img, (p[i], p[i+5]), 3, (0,255,0), 1)
    return img

def write_text(img, text):
    cv2.putText(img, text, (5,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255,0), lineType=cv2.LINE_AA)
    return img
