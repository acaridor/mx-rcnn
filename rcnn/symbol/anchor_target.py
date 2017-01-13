"""
AnchorTarget Operator select anchors from a dense grid, assign label and bbox transform to them.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr
from distutils.util import strtobool

from rcnn.io.rpn import assign_anchor
from rcnn.io.image import tensor_vstack

DEBUG = False


class AnchorTargetOperator(mx.operator.CustomOp):
    def __init__(self, feat_stride, scales, ratios, allowed_border):
        super(AnchorTargetOperator, self).__init__()
        self._feat_stride = feat_stride
        self._scales = scales
        self._ratios = ratios
        self._allowed_border = allowed_border

    def forward(self, is_train, req, in_data, out_data, aux):
        feat_shape = in_data[0].shape
        batch_size = feat_shape[0]
        batch_gt_boxes = in_data[1].asnumpy()
        batch_im_info = in_data[2].asnumpy()

        labels = []
        bbox_targets = []
        bbox_weights = []
        for i in range(batch_size):
            gt_boxes = batch_gt_boxes[i]
            im_info = batch_im_info[i]
            label, bbox_target, bbox_weight = \
                assign_anchor(feat_shape, gt_boxes, im_info,
                              self._feat_stride, self._scales, self._ratios, self._allowed_border)
            labels.append(label)
            bbox_targets.append(bbox_target)
            bbox_weights.append(bbox_weight)

        for ind, val in enumerate([labels, bbox_targets, bbox_weights]):
            val = tensor_vstack(val)
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for ind in range(len(in_grad)):
            self.assign(in_grad[ind], req[ind], 0)


@mx.operator.register("anchor_target")
class AnchorTargetProp(mx.operator.CustomOpProp):
    def __init__(self, feat_stride='16', scales='(8, 16, 32)', ratios='(0.5, 1, 2)', allowed_border='0'):
        super(AnchorTargetProp, self).__init__(need_top_grad=False)
        self._feat_stride = int(feat_stride)
        self._scales = np.fromstring(scales[1:-1], dtype=float, sep=',')
        self._ratios = np.fromstring(ratios[1:-1], dtype=float, sep=',')
        self._num_anchors = len(self._scales) * len(self._ratios)
        self._allowed_border = int(allowed_border)

    def list_arguments(self):
        return ['conv_feat', 'gt_boxes', 'im_info']

    def list_outputs(self):
        return ['label', 'bbox_target', 'bbox_weight']

    def infer_shape(self, in_shape):
        conv_feat_shape = in_shape[0]
        batch_size = int(conv_feat_shape[0])
        height = int(conv_feat_shape[2])
        width = int(conv_feat_shape[3])

        label_shape = (batch_size, self._num_anchors * height * width)
        bbox_target_shape = (batch_size, self._num_anchors * 4, height, width)
        bbox_weight_shape = (batch_size, self._num_anchors * 4, height, width)

        return in_shape, [label_shape, bbox_target_shape, bbox_weight_shape]

    def create_operator(self, ctxs, shapes, dtypes):
        return AnchorTargetOperator(self._feat_stride, self._scales, self._ratios, self._allowed_border)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
