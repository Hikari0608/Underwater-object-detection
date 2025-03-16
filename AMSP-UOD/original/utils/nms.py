import torch
from numpy import array
from utils.general import box_iou
import numpy as np

def np_box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clip(0).prod(2)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) -
            np.maximum(box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)

def np_nms(boxes :array, scores :array, iou_threshold :float):
    boxes = np.array(boxes.to('cpu'))
    scores = np.array(scores.to('cpu'))
    idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
    keep = []
    while idxs.size > 0:  # 统计数组中元素的个数
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = np_box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]
    keep = np.array(keep)  
    return keep


def torch_soft_nms(boxes :array, scores :array, iou_threshold :float, sigma=0.3, soft_threshold=0.1):
    keep = []
    idxs = scores.argsort()
    while idxs.numel() > 0:
        idxs = scores.argsort() # 得分变化
        if idxs.size(0) == 1:
            keep.append(idxs[-1])
            break
        keep_len = len(keep)
        max_score_index = idxs[-(keep_len + 1)]
        max_score_box = boxes[max_score_index][None, :]
        idxs = idxs[:-(keep_len + 1)]
        other_boxes = boxes[idxs] 
        keep.append(max_score_index)
        ious = box_iou(max_score_box, other_boxes) 
        decay = torch.exp(-(ious[0] ** 2) / sigma)
        scores[idxs] *= decay
    keep = idxs.new(keep)  # Tensor
    keep = keep[scores[keep] > soft_threshold]
    return keep

def nms_s(boxes, scores, iou_threshold, sigma=0.3, soft_threshold=0.1, aspect_threshold=0.8):
    device = boxes.device
    keep = []

    # Pre-sort scores and boxes only once
    idxs = scores.argsort(descending=True)

    while idxs.numel() > 0:
        max_score_index = idxs[0]
        keep.append(max_score_index)

        if idxs.numel() == 1:
            break

        max_score_box = boxes[max_score_index].unsqueeze(0)
        other_boxes = boxes[idxs[1:]]

        ious = box_iou(max_score_box, other_boxes)
        asps = box_asp(max_score_box, other_boxes)

        # Soft NMS decay
        decay = torch.exp(-(ious[0] ** 2) / sigma)
        scores[idxs[1:]] *= decay

        aspect_filter = asps >= aspect_threshold
        idxs = idxs[1:][aspect_filter & (ious[0] <= iou_threshold)]


    # Convert to Tensor and apply final threshold
    keep = torch.tensor(keep).to(device)
    keep = keep[scores[keep] > soft_threshold]

    return keep

def np_nms_bk(boxes :array, scores :array, iou_threshold :float, sigma=0.3, soft_threshold=0.1, aspect_threshold=0.98):
    keep = []
     # 值从小到大的 索引， 索引对应的 是 元boxs索引 scores索引
    idxs = scores.argsort()
    while idxs.numel() > 0:  # 循环直到null； numel()： 数组元素个数
        # 得分最大框对应的索引, 以及对应的坐标
        # 由于scores得分会改变，所以每次都要重新排序，获取得分最大值
        idxs = scores.argsort()  # 评分排序

        if idxs.size(0) == 1:  # 就剩余一个框了；
            keep.append(idxs[-1])  # 位置不能边
            break
        keep_len = len(keep)
        max_score_index = idxs[-(keep_len + 1)]
        max_score_box = boxes[max_score_index][None, :]  # [1, 4]
        idxs = idxs[:-(keep_len + 1)]
        other_boxes = boxes[idxs]  # [?, 4]
        keep.append(max_score_index)  # 位置不能边
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        # Soft NMS 处理， 和 得分最大框 IOU大于阈值的框， 进行得分抑制
        decay = torch.exp(-(ious[0] ** 2) / sigma)
        scores[idxs] *= decay
        asps = box_asp(max_score_box, other_boxes)
        aspect_filter = asps > aspect_threshold
        idxs = idxs[(aspect_filter | (ious[0] <= iou_threshold))]
    keep = idxs.new(keep)  # Tensor
    keep = keep[scores[keep] > soft_threshold]  # 最后处理阈值
    return keep

def box_asp_np(box1, box2, eps=1e-7):
    # 将pytorch张量转换为numpy数组，并断开与计算图的连接 
    (a1, a2), (b1, b2) = np.split(box1[np.newaxis, :], 2, axis=2), np.split(box2[:, np.newaxis, :], 2, axis=2) 
    a, b = a2 - a1 + eps, b2 - b1 + eps # 使用numpy.linalg.norm和numpy.dot方法来计算两个数组的范数和点积 
    a_norm = np.linalg.norm(a, axis=2)
    b_norm = np.linalg.norm(b, axis=2) 
    dot_ab = np.sum(a * b, axis=2) # 使用点积除以范数乘积来计算两个数组的余弦相似度 
    return (dot_ab / (a_norm * b_norm))

def box_asp(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    a, b = a2 - a1 + eps, b2 - b1 + eps
    return torch.cosine_similarity(a,b,dim=2).squeeze(0)