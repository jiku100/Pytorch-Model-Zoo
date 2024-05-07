from BaseLoader import BaseLoader
import torch
import torch.nn as nn
import numpy as np
from time import time
import cv2
from typing import Union
import sys
sys.path.append("./Yet-Another-EfficientDet-Pytorch")

from backbone import EfficientDetBackbone
from torchvision.ops.boxes import batched_nms

class BBoxTransform(nn.Module):
    def forward(self, anchors, regression):
        """
        decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

        Args:
            anchors: [batchsize, boxes, (y1, x1, y2, x2)]
            regression: [batchsize, boxes, (dy, dx, dh, dw)]

        Returns:

        """
        y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
        x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
        ha = anchors[..., 2] - anchors[..., 0]
        wa = anchors[..., 3] - anchors[..., 1]

        w = regression[..., 3].exp() * wa
        h = regression[..., 2].exp() * ha

        y_centers = regression[..., 0] * ha + y_centers_a
        x_centers = regression[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class ClipBoxes(nn.Module):

    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)

        return boxes


class EfficientDetLoader(BaseLoader):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        self.anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

        self.compound_coef = None

        self.model = self.load()

        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        
        
        
    def load(self):
        if self.model_name.startswith("efficientdet"):
            self.compound_coef = int(self.model_name[-1])

            try:
                model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=90,
                             ratios=self.anchor_ratios, scales=self.anchor_scales)
                model.load_state_dict(torch.load(self.model_path))
                model.requires_grad_(False)
                model.eval()

                if self.device == "cuda":
                    model = model.cuda()
                
            except Exception as e:
                raise e
            
        return model
    
    def warmup(self):
        inputs = []
        for _ in range(self.batch_size):
            input = np.random.randint(low=0, high=255, size=(self.resolution[1], self.resolution[0], 3), dtype='uint8')
            inputs.append(input)

        ori_imgs, framed_imgs, framed_metas = self.preprocess(inputs)

        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        x = x.to(torch.float32).permute(0, 3, 1, 2)

        with torch.no_grad():
            for _ in range(10):
                _, regression, classification, anchors = self.model(x)

                out = self.postprocess(x,
                                anchors, regression, classification,
                                self.regressBoxes, self.clipBoxes)
                out = self.invert_affine(framed_metas, out)

    def benchmark(self):
        print("Benchmarking EfficientDet")
        print(f"Configs: {self.model_name} {self.resolution} {self.batch_size}")

        self.warmup()

        inputs = []
        for _ in range(self.batch_size):
            input = np.random.randint(low=0, high=255, size=(self.resolution[1], self.resolution[0], 3), dtype='uint8')
            inputs.append(input)

        start = time()
        ori_imgs, framed_imgs, framed_metas = self.preprocess(inputs)

        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        x = x.to(torch.float32).permute(0, 3, 1, 2)

        with torch.no_grad():
            for _ in range(10):
                _, regression, classification, anchors = self.model(x)

                out = self.postprocess(x,
                                anchors, regression, classification,
                                self.regressBoxes, self.clipBoxes)
                out = self.invert_affine(framed_metas, out)
    
        end = time()

        average_inference_time = (end - start) / 10

        print(f"Latency: {average_inference_time * 1000:.4f} milliseconds\n")

        return average_inference_time
    
    def invert_affine(self, metas: Union[float, list, tuple], preds):
        for i in range(len(preds)):
            if len(preds[i]['rois']) == 0:
                continue
            else:
                if metas is float:
                    preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                    preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
                else:
                    new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                    preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                    preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
        return preds


    def aspectaware_resize_padding(self, image, width, height, interpolation=None, means=None):
        old_h, old_w, c = image.shape
        if old_w > old_h:
            new_w = width
            new_h = int(width / old_w * old_h)
        else:
            new_w = int(height / old_h * old_w)
            new_h = height

        canvas = np.zeros((height, height, c), np.float32)
        if means is not None:
            canvas[...] = means

        if new_w != old_w or new_h != old_h:
            if interpolation is None:
                image = cv2.resize(image, (new_w, new_h))
            else:
                image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        padding_h = height - new_h
        padding_w = width - new_w

        if c > 1:
            canvas[:new_h, :new_w] = image
        else:
            if len(image.shape) == 2:
                canvas[:new_h, :new_w, 0] = image
            else:
                canvas[:new_h, :new_w] = image

        return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h,

    def preprocess(self, ori_imgs, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        normalized_imgs = [(img[..., ::-1] / 255 - mean) / std for img in ori_imgs]
        imgs_meta = [self.aspectaware_resize_padding(img, self.resolution[0], self.resolution[0],
                                                means=None) for img in normalized_imgs]
        framed_imgs = [img_meta[0] for img_meta in imgs_meta]
        framed_metas = [img_meta[1:] for img_meta in imgs_meta]

        return ori_imgs, framed_imgs, framed_metas

    def postprocess(self, x, anchors, regression, classification, regressBoxes, clipBoxes, threshold=0.2, iou_threshold=0.2):
        transformed_anchors = regressBoxes(anchors, regression)
        transformed_anchors = clipBoxes(transformed_anchors, x)
        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > threshold)[:, :, 0]
        out = []
        for i in range(x.shape[0]):
            if scores_over_thresh[i].sum() == 0:
                out.append({
                    'rois': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                })
                continue

            classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
            transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
            scores_per = scores[i, scores_over_thresh[i, :], ...]
            scores_, classes_ = classification_per.max(dim=0)
            anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

            if anchors_nms_idx.shape[0] != 0:
                classes_ = classes_[anchors_nms_idx]
                scores_ = scores_[anchors_nms_idx]
                boxes_ = transformed_anchors_per[anchors_nms_idx, :]

                out.append({
                    'rois': boxes_.cpu().numpy(),
                    'class_ids': classes_.cpu().numpy(),
                    'scores': scores_.cpu().numpy(),
                })
            else:
                out.append({
                    'rois': np.array(()),
                    'class_ids': np.array(()),
                    'scores': np.array(()),
                })

        return out