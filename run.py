import os
import pickle
import sys
sys.path.append("./Yet_Another_EfficientDet_Pytorch")

from YoloLoader import YoloLoader
from EfficientDetLoader import EfficientDetLoader
from configs import RESOLUTIONS

def YoloTest(output_dir: str):
    models = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
    batch_sizes = [1, 2, 4, 8, 16]
    
    benchmark_output_path = os.path.join(output_dir, "yolo.pkl")

    results = {}

    for model in models:
        yolo = YoloLoader(model)

        result = {}

        for resolution in RESOLUTIONS:
            latencies = []

            yolo.setResolution(resolution[0], resolution[1])

            for batch_size in batch_sizes:
                yolo.setBatchSize(batch_size)
                latency = yolo.benchmark()
                latencies.append(latency)
            
            result[resolution[1]] = latencies
        
        results[model] = result
    
    with open(benchmark_output_path, "wb") as f:
        pickle.dump(results, f)

def EfficientDetTest(output_dir: str):
    models = ['efficientdet-d0', 'efficientdet-d1', 'efficientdet-d2', 'efficientdet-d3', 'efficientdet-d4', 'efficientdet-d5', 'efficientdet-d6', 'efficientdet-d7']
    batch_sizes = [1, 2, 4, 8, 16]

    benchmark_output_path = os.path.join(output_dir, "efficientdet.pkl")

    results = {}

    for model in models:
        efficientdet = EfficientDetLoader(model)

        result = {}

        for resolution in [RESOLUTIONS[0], RESOLUTIONS[2]]:
            latencies = []

            efficientdet.setResolution(resolution[0], resolution[1])

            for batch_size in batch_sizes:
                efficientdet.setBatchSize(batch_size)
                latency = efficientdet.benchmark()

            result[resolution[1]] = latencies
        
        results[model] = result

    with open(benchmark_output_path, "wb") as f:
        pickle.dump(results, f)

def main():
    benchmark_output_dir = "benchmark"
    if not os.path.exists(benchmark_output_dir):
        os.makedirs(benchmark_output_dir)

    YoloTest(benchmark_output_dir)
    EfficientDetTest(benchmark_output_dir)


if __name__ == "__main__":
    main()

