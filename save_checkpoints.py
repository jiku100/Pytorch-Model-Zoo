import torch
import os
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2,
    efficientnet_b3, efficientnet_b4, efficientnet_b5,
    efficientnet_b6, efficientnet_b7
)

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


# # 각 EfficientNet 버전을 함수로 매핑
# efficientnet_versions = {
#     "efficientnet-b0": efficientnet_b0,
#     "efficientnet-b1": efficientnet_b1,
#     "efficientnet-b2": efficientnet_b2,
#     "efficientnet-b3": efficientnet_b3,
#     "efficientnet-b4": efficientnet_b4,
#     "efficientnet-b5": efficientnet_b5,
#     "efficientnet-b6": efficientnet_b6,
#     "efficientnet-b7": efficientnet_b7,
# }

# # 가중치를 저장할 기본 경로 지정
# base_path = 'checkpoints/efficientnet'

# os.makedirs(base_path, exist_ok=True)

# # 각 모델의 가중치 저장 및 로드
# for model_name, model_fn in efficientnet_versions.items():
#     # 모델 로드
#     model = model_fn(pretrained=True)
    
#     # 가중치 저장 경로 설정
#     weight_path = f'{base_path}/{model_name}.pth'
    
#     # 가중치 저장
#     torch.save(model.state_dict(), weight_path)
    
#     # 가중치 로드를 위한 모델 초기화 (실제 필요한 경우에만 사용)
#     model = model_fn(pretrained=False)
#     model.load_state_dict(torch.load(weight_path))

#     print(f'Model {model_name} weights saved and reloaded successfully from {weight_path}')

# 각 ResNet 버전을 함수로 매핑
resnet_versions = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}

# 가중치를 저장할 기본 경로 지정
base_path = 'checkpoints/resnet'

os.makedirs(base_path, exist_ok=True)

# 각 모델의 가중치 저장 및 로드
for model_name, model_fn in resnet_versions.items():
    # 모델 로드
    model = model_fn(pretrained=True)
    
    # 가중치 저장 경로 설정
    weight_path = f'{base_path}/{model_name}.pth'
    
    # 가중치 저장
    torch.save(model.state_dict(), weight_path)
    
    # 가중치 로드를 위한 모델 초기화 (실제 필요한 경우에만 사용)
    model = model_fn(pretrained=False)
    model.load_state_dict(torch.load(weight_path))

    print(f'Model {model_name} weights saved and reloaded successfully from {weight_path}')