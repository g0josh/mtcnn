import math
import numpy as np
import cv2
import torch
from .model import PNet, RNet, ONet
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from .box_utils import _convert_to_square, _preprocess, _nms, _calibrate_box, _get_image_boxes

def detect_faces(image, min_face_size=20.0, thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pnet, rnet, onet= PNet(), RNet(), ONet()
    onet.eval()

    height = image.shape[0]
    width = image.shape[1]
    # width, height = image.size
    min_length = min(height, width)
    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    scales = []
    m = min_detection_size/min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m*factor**factor_count)
        min_length *= factor
        factor_count += 1

    # convert cv2 image to torch
    image = _preprocess(torch.FloatTensor(image))

    # STAGE 1
    bounding_boxes = []
    for s in scales:    # run P-Net on different scales
        boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0], device=device)
        if boxes is not None and boxes.numel() > 0:
            bounding_boxes.append(boxes)
    bounding_boxes = torch.cat(bounding_boxes, dim=0)

    keep = _nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = _calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    bounding_boxes = _convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = torch.round(bounding_boxes[:, 0:4])
    # bounding_boxes = bounding_boxes.data.numpy()

    # STAGE 2
    img_boxes = _get_image_boxes(bounding_boxes, image, size=24)
    # img_boxes = torch.FloatTensor(img_boxes)
    output = rnet(img_boxes)
    offsets = output[0]  # shape [n_boxes, 4]
    probs = output[1] # shape [n_boxes, 2]

    # bounding_boxes = bounding_boxes.data.numpy()
    keep = torch.nonzero(probs[:, -1] > thresholds[1])
    # keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].view((-1,1))
    offsets = offsets[keep]

    keep = _nms(bounding_boxes, nms_thresholds[1])
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = _calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = _convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = torch.round(bounding_boxes[:, 0:4])

    # STAGE 3
    img_boxes = _get_image_boxes(bounding_boxes, image, size=48)
    if img_boxes.numel() == 0:
        return [], []
    # img_boxes = torch.FloatTensor(img_boxes)
    output = onet(img_boxes)
    landmarks = output[0]  # shape [n_boxes, 10]
    offsets = output[1]  # shape [n_boxes, 4]
    probs = output[2]  # shape [n_boxes, 2]

    keep = torch.non_zero(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]

    return bounding_boxes, landmarks

def run_first_stage(image, net, scale, threshold, device):
    """
        Run P-Net, generate bounding boxes, and do NMS.
    """
    # img = cv2.resize(image, (int(scale*image.shape[1]), int(scale*image.shape[0])) )
    # print "cv2", scale, img.shape
    # _img = _preprocess(torch.FloatTensor(image))
    _cropped = torch.nn.functional.upsample(_img, scale_factor=scale, mode='bilinear')
    # print "torch", scale, _cropped.shape
    # img = _preprocess(torch.FloatTensor(img))

    output = net(img)
    probs = output[1][0,1,:,:]
    offsets = output[0]
    # probs = output[1].data.numpy()[0, 1, :, :]
    # offsets = output[0].data.numpy()
    # print (output[0].shape, output[1].shape, probs.shape, offsets.shape)
    # print("probs = {}\n".format(probs))

    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if boxes is None or boxes.numel() == 0:
        return None

    # boxes = boxes.data.numpy()
    keep = _nms(boxes[:, 0:5], overlap_threshold=0.5)
    return boxes[keep]

def _generate_bboxes(probs, offsets, scale, threshold):
    """
       Generate bounding boxes at places where there is probably a face.
    """
    stride = 2
    cell_size = 12

    inds = torch.nonzero(probs > threshold)

    if inds.numel() == 0:
        return None

    offsets = offsets[:, :, inds[:,0], inds[:,1]].squeeze(0)
    score = probs[inds[:,0], inds[:,1]]
    inds = inds.to(dtype=torch.float)
    # print ("inds = {}/{}, offsets = {}/{}".format(inds.shape,inds.dtype,offsets.shape,offsets.dtype))

    # P-Net is applied to scaled images, so we need to rescale bounding boxes back
    bounding_boxes = torch.cat([
        torch.round((stride*inds[:,1] + 1.0)/scale).unsqueeze(0),
        torch.round((stride*inds[:,0] + 1.0)/scale).unsqueeze(0),
        torch.round((stride*inds[:,1] + 1.0 + cell_size)/scale).unsqueeze(0),
        torch.round((stride*inds[:,0] + 1.0 + cell_size)/scale).unsqueeze(0),
        score.unsqueeze(0),
        offsets
    ])
    # print ("bb = {}".format(bounding_boxes.shape))

    return bounding_boxes.transpose(1,0)
