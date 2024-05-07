import os
from models import MODELS

class BaseLoader:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = MODELS.get(model_name)

        base_dir = os.path.dirname(self.model_path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        self.model = None

        self.resolution = [640, 480] ## Width, Height
        self.batch_size = 1

    def load(self):
        pass

    def warmup(self):
        pass

    def forward(self):
        pass

    def benchmark(self):
        pass

    def setResolution(self, width: int, height: int):
        self.resolution = [width, height]
        return self

    def setBatchSize(self, batch_size: int):
        self.batch_size = batch_size
        return self

