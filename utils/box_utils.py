import torch
import numpy as np

def nms(boxes, overlap_threshold=0.5, min_mode=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        keep.append(order[0])
        xx1 = np.maximum(x1[order[0]], x1[order[1:]])
        yy1 = np.maximum(y1[order[0]], y1[order[1:]])
        xx2 = np.minimum(x2[order[0]], x2[order[1:]])
        yy2 = np.minimum(y2[order[0]], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if min_mode:
            ovr = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            ovr = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]
    return keep

def _nms(boxes, overlap_threshold=0.5, mode='union'):
    # This native torch implementation is slow
    # on cuda for cuda tensors
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

    cropped = torch.zeros((num_bboxes, 3, size, size), dtype=img.dtype, device=img.device)
    for i in range(num_bboxes):
        bbox = bboxes_c[i]
        _cropped = img[:, : ,bbox[1]:bbox[3], bbox[0]:bbox[2]]
        cropped[i,:,:,:] = torch.nn.functional.interpolate(_cropped, size=size, mode='bilinear')

    return cropped

def correct_bboxes(bboxes, img):
    # all bbox dims to be within the image

    imgh = img.shape[-2]
    imgw = img.shape[-1]

    # bboxes_c = torch.zeros_like(bboxes, dtype = torch.int)
    bboxes[:,:2] = bboxes[:,:2].clamp(min=0)
    bboxes[:,2] = bboxes[:,2].clamp(max=imgw-1)
    bboxes[:,3] = bboxes[:,3].clamp(max=imgh-1)
    return bboxes.to(torch.int)

def cv2Torch(image, device):
    """Preprocessing step before feeding the network.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = torch.tensor(image, dtype=torch.float, device=device)
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    img = (img - 127.5)*0.0078125
    return img