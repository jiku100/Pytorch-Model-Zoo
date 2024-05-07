from BaseLoader import BaseLoader
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7 
from PIL import Image
import numpy as np
from time import time

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class EfficientNetLoader(BaseLoader):
    def __init__(self, model_name: str):
        super().__init__(model_name)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = self.load()

    def load(self):
        efficientnet_models = {
            "efficientnet-b0": efficientnet_b0,
            "efficientnet-b1": efficientnet_b1,
            "efficientnet-b2": efficientnet_b2,
            "efficientnet-b3": efficientnet_b3,
            "efficientnet-b4": efficientnet_b4,
            "efficientnet-b5": efficientnet_b5,
            "efficientnet-b6": efficientnet_b6,
            "efficientnet-b7": efficientnet_b7,
        }
        try:
            model = efficientnet_models[self.model_name](pretrained=False)
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
        print("Benchmarking EfficientNet")
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