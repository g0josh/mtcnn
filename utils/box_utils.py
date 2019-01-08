import torch

def _nms(boxes, overlap_threshold=0.5, mode='union'):
    import numpy as np
    boxes = boxes.data.numpy()
    """ Pure Python NMS baseline. """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if mode is 'min':
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]

    return keep


def nms(boxes, overlap_threshold=0.5, mode='union'):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    _, order = scores.sort(dim=0, descending=True)
    ind_buffer = torch.zeros(scores.shape, dtype=torch.long)
    i = 0
    while order.size()[0] > 1:
        ind_buffer[i] = order[0]
        i += 1
        xx1 = torch.max(x1[order[0]], x1[order[1:]])
        yy1 = torch.max(y1[order[0]], y1[order[1:]])
        xx2 = torch.min(x2[order[0]], x2[order[1:]])
        yy2 = torch.min(y2[order[0]], y2[order[1:]])

        # w = F.relu(xx2 - xx1)
        # h = F.relu(yy2 - yy1)
        w = torch.clamp(xx2 - xx1 + 1, min=0)
        h = torch.clamp(yy2 - yy1 + 1, min=0)
        inter = w * h
        if mode == 'min':
            ovr = inter / torch.min(areas[order[0]], areas[order[1:]])
        else:
            ovr = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = torch.nonzero(ovr <= overlap_threshold).squeeze()
        if inds.dim():
            order = order[(inds + 1)]
        else:
            break
    keep = ind_buffer[:i]
    return keep

def convert_to_square(bboxes):
    """
        Convert bounding boxes to a square form.
    """
    square_bboxes = torch.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = torch.max(h, w)
    square_bboxes[:, 0] = x1 + w*0.5 - max_side*0.5
    square_bboxes[:, 1] = y1 + h*0.5 - max_side*0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes

def calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
    'offsets' is one of the outputs of the nets.
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = w.unsqueeze(1)
    h = h.unsqueeze(1)

    translation = torch.cat([w, h, w, h], 1) * offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes

def get_image_boxes(bboxes, img, size=24):

    bboxes_c = correct_bboxes(bboxes, img)
    num_bboxes = bboxes_c.shape[0]

    cropped = []
    for i in range(num_bboxes):
        bbox = bboxes_c[i]
        _cropped = img[:, : ,bbox[1]:bbox[3], bbox[0]:bbox[2]]
        _cropped = torch.nn.functional.interpolate(_cropped, size=size, mode='bilinear')
        cropped.append(_cropped)

    return torch.cat(cropped)

def correct_bboxes(bboxes, img):
    # all bbox dims to be within the image

    imgh = img.shape[-2]
    imgw = img.shape[-1]

    x1 = bboxes[:,0]
    x1_c = torch.clamp(x1, min = 0).unsqueeze(0)

    y1 = bboxes[:,1]
    y1_c = torch.clamp(y1, min = 0).unsqueeze(0)

    x2 = bboxes[:,2]
    x2_c = torch.clamp(x2, max = imgw-1).unsqueeze(0)

    y2 = bboxes[:,3]
    y2_c = torch.clamp(y2, max = imgh-1).unsqueeze(0)

    bboxes_c = torch.cat([x1_c,y1_c,x2_c,y2_c]).transpose(1,0)

    return bboxes_c.to(dtype=torch.int)

def preprocess(img):
    """Preprocessing step before feeding the network.
    """
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    img = (img - 127.5)*0.0078125
    return img