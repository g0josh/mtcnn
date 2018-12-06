import torch


def _nms(boxes, overlap_threshold=0.5, mode='union'):
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

def nms(boxes, overlap_threshold=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap_threshold: (float) The overlap threshold for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    keep = scores.new(scores.size(0)).zero_().long()
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        xx1=torch.index_select(x1, 0, idx)
        yy1=torch.index_select(y1, 0, idx)
        xx2=torch.index_select(x2, 0, idx)
        yy2=torch.index_select(y2, 0, idx)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i].data)
        yy1 = torch.clamp(yy1, min=y1[i].data)
        xx2 = torch.clamp(xx2, max=x2[i].data)
        yy2 = torch.clamp(yy2, max=y2[i].data)
        # w.resize_as_(xx2)
        # h.resize_as_(yy2)
        w = xx2 - xx1 +1
        h = yy2 - yy1 +1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap_threshold
        idx = idx[IoU.le(overlap_threshold)]
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