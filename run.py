import os
from YoloLoader import YoloLoader

def main():
    # yolo = YoloLoader("yolov8n")
    # yolo.setBatchSize(16)
    # yolo.benchmark()

    import torch
    entrypoints = torch.hub.list('rwightman/efficientdet-pytorch', force_reload=True)
    print(entrypoints)

    # # Torch Hub에서 EfficientDet 모델 로드
    # model = torch.hub.load('rwightman/efficientdet-pytorch', 'tf_efficientdet_d1', pretrained=True)

    # # 모델을 평가 모드로 설정
    # model.eval()

    # 모델 사용 예제
    # 예를 들어, 이미지를 모델에 입력하여 결과를 받아볼 수 있습니다.



if __name__ == "__main__":
    main()

