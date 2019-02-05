import numpy as np
cimport numpy as np

cpdef np.ndarray[np.int_t, ndim=1] nms_cpu(np.ndarray[np.float32_t, ndim=2] boxes,
        float overlap_threshold=0.5,
        bint min_mode=False):
    cdef np.ndarray[np.float32_t, ndim=1] x1 = boxes[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = boxes[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = boxes[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = boxes[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = boxes[:, 4]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]
    cdef np.ndarray[np.int_t, ndim=1] keep = np.zeros(scores.shape[0], dtype=np.int)

    cdef int i = 0
    cdef int cnt = 0
    cdef np.ndarray[np.float32_t, ndim=1] xx1, xx2, yy1, yy2
    cdef np.ndarray[np.float32_t, ndim=1] w, h, inter, ovr
    cdef np.ndarray[np.int_t, ndim=1] inds

    while order.size > 0:
        i = order[0]
        keep[cnt] = i
        cnt += 1
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if min_mode:
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        else:
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]

    return keep[:cnt]