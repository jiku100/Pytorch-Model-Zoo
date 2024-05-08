import os
import pickle
import sys
sys.path.append("./Yet_Another_EfficientDet_Pytorch")

from YoloLoader import YoloLoader
from EfficientDetLoader import EfficientDetLoader
from EfficientNetLoader import EfficientNetLoader
from ResNetLoader import ResNetLoader

from configs import RESOLUTIONS

def YoloTest(output_dir: str):
    models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
    batch_size = 1

    benchmark_output_path = os.path.join(output_dir, "yolo.pkl")

    results = {}

    resolutions = RESOLUTIONS
    
    results["resolutions"] = [resolution[1] for resolution in resolutions]

    for model in models:
        yolo = YoloLoader(model)
        yolo.setBatchSize(batch_size)

        latencies = []
        for resolution in resolutions:
            
            yolo.setResolution(resolution[0], resolution[1])
            latency = yolo.benchmark()
            latencies.append(latency)
      
        results[model] = latencies
    
    with open(benchmark_output_path, "wb") as f:
        pickle.dump(results, f)

def EfficientDetTest(output_dir: str):
    models = ['efficientdet-d0', 'efficientdet-d1', 'efficientdet-d2', 'efficientdet-d3', 'efficientdet-d4', 'efficientdet-d5', 'efficientdet-d6', 'efficientdet-d7']
    batch_size = 1

    benchmark_output_path = os.path.join(output_dir, "efficientdet.pkl")

    results = {}

    resolutions = [RESOLUTIONS[0], RESOLUTIONS[2]]
    
    results["resolutions"] = [resolution[1] for resolution in resolutions]

    for model in models:
        efficientdet = EfficientDetLoader(model)
        efficientdet.setBatchSize(batch_size)

        latencies = []
        for resolution in resolutions:
            
            efficientdet.setResolution(resolution[0], resolution[1])                
            latency = efficientdet.benchmark()
            latencies.append(latency)
        
        results[model] = latencies

    with open(benchmark_output_path, "wb") as f:
        pickle.dump(results, f)

def EfficientNetTest(output_dir: str):
    models = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']
    batch_sizes = [1, 2, 4, 8, 16]

    benchmark_output_path = os.path.join(output_dir, "efficientnet.pkl")

    results = {}
    results['batch_sizes'] = batch_sizes

    for model in models:
        efficientnet = EfficientNetLoader(model)
        efficientnet.setResolution(224, 224)

        latencies = []

        for batch_size in batch_sizes:
            efficientnet.setBatchSize(batch_size)
            latency = efficientnet.benchmark()
            latencies.append(latency)
        
        results[model] = latencies

    with open(benchmark_output_path, "wb") as f:
        pickle.dump(results, f)

def ResNetTest(output_dir: str):
    models = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    batch_sizes = [1, 2, 4, 8, 16]

    benchmark_output_path = os.path.join(output_dir, "resnet.pkl")

    results = {}
    results['batch_sizes'] = batch_sizes
    
    for model in models:
        resnet = ResNetLoader(model)
        resnet.setResolution(224, 224)

        latencies = []

        for batch_size in batch_sizes:
            resnet.setBatchSize(batch_size)
            latency = resnet.benchmark()
            latencies.append(latency)

        results[model] = latencies

    with open(benchmark_output_path, "wb") as f:
        pickle.dump(results, f)

def main():
    benchmark_output_dir = "benchmark"
    if not os.path.exists(benchmark_output_dir):
        os.makedirs(benchmark_output_dir)

    YoloTest(benchmark_output_dir)
    EfficientDetTest(benchmark_output_dir)

    EfficientNetTest(benchmark_output_dir)
    ResNetTest(benchmark_output_dir)

if __name__ == "__main__":
    main()

