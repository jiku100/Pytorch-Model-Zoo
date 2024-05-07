import os
import sys
sys.path.append("./Yet_Another_EfficientDet_Pytorch")

from YoloLoader import YoloLoader
from EfficientDetLoader import EfficientDetLoader

def YoloTest():
    models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
    batch_sizes = [1, 2, 4, 8, 16]

    for model in models:
        yolo = YoloLoader(model)

        for batch_size in batch_sizes:
            yolo.setBatchSize(batch_size)
            yolo.benchmark()

def EfficientDetTest():
    models = ['efficientdet-d0', 'efficientdet-d1', 'efficientdet-d2', 'efficientdet-d3', 'efficientdet-d4', 'efficientdet-d5', 'efficientdet-d6', 'efficientdet-d7']
    batch_sizes = [1, 2, 4, 8, 16]

    for model in models:
        efficientdet = EfficientDetLoader(model)

        for batch_size in batch_sizes:
            efficientdet.setBatchSize(batch_size)
            efficientdet.benchmark()

def main():
    EfficientDetTest()


if __name__ == "__main__":
    main()

