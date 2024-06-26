
import numpy as np
import os

from time import time 
import torch
from BaseLoader import BaseLoader
from ultralytics import YOLO

class YoloLoader(BaseLoader):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        self.model = self.load()

    def load(self):
        try:
            model = YOLO(self.model_path).to(self.device)
            
        except Exception as e:
            raise e
            
        return model

    def warmup(self):
        inputs = []
        for _ in range(self.batch_size):
            input = np.random.randint(low=0, high=255, size=(self.resolution[1], self.resolution[0], 3), dtype='uint8')
            inputs.append(input)

        for _ in range(10):
            self.model.predict(inputs, verbose=False)
    
    def benchmark(self):
        print("Benchmarking YOLOv8")
        print(f"Configs: {self.model_name} {self.resolution} {self.batch_size}")

        self.warmup()

        inputs = []
        for _ in range(self.batch_size):
            input = np.random.randint(low=0, high=255, size=(self.resolution[1], self.resolution[0], 3), dtype='uint8')
            inputs.append(input)

        start = time()
        for _ in range(10):
            self.model.predict(inputs, verbose=False)
        end = time()

        average_inference_time = (end - start) / 10

        print(f"Latency: {average_inference_time * 1000:.4f} milliseconds\n")

        return average_inference_time * 1000
    