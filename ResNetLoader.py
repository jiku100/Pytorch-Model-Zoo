from BaseLoader import BaseLoader
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from PIL import Image
import numpy as np
from time import time

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ResNetLoader(BaseLoader):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = self.load()

    def load(self):
        resnet_models = {
            "resnet18": resnet18,
            "resnet34": resnet34,
            "resnet50": resnet50,
            "resnet101": resnet101,
            "resnet152": resnet152
        }

        try:
            model = resnet_models[self.model_name](pretrained=False)
            model.load_state_dict(torch.load(self.model_path))
            model = model.to(self.device).eval()
        except Exception as e:
            print(f"Failed to load the model {self.model_name}. Error: {str(e)}")
            raise e
        
        return model

    def warmup(self):
        inputs = []
        for _ in range(self.batch_size):
            input = transform(Image.fromarray(np.random.randint(low=0, high=255, size=(224, 224, 3), dtype='uint8')))
            inputs.append(input)

        x = torch.stack(inputs).to(self.device)

        with torch.no_grad():
            for _ in range(10):
                self.model(x)

    def benchmark(self):
        print("Benchmarking ResNet")
        print(f"Configs: {self.model_name} {self.resolution} {self.batch_size}")

        self.warmup()

        inputs = []
        for _ in range(self.batch_size):
            input = transform(Image.fromarray(np.random.randint(low=0, high=255, size=(224, 224, 3), dtype='uint8')))
            inputs.append(input)

        start = time()

        x = torch.stack(inputs).to(self.device)

        with torch.no_grad():
            for _ in range(10):
                self.model(x)
    
        end = time()

        average_inference_time = (end - start) / 10

        print(f"Latency: {average_inference_time * 1000:.4f} milliseconds\n")

        return average_inference_time * 1000