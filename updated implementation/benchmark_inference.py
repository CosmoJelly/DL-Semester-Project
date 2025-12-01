# benchmark_inference.py
"""
Benchmark inference latency and memory usage for all models.
Useful for deployment and efficiency analysis.
"""
import os, time, argparse, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from model_builder import build_model, get_base_model
from config import IMG_SIZE, SUPPORTED_MODELS, MODELS_DIR, RESULTS_DIR

def get_gpu_memory_info():
    """Get current GPU memory usage."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Enable memory growth to get accurate measurements
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Get memory info
            memory_info = {}
            for i, gpu in enumerate(gpus):
                try:
                    details = tf.config.experimental.get_device_details(gpu)
                    memory_info[f'gpu_{i}'] = {
                        'name': details.get('device_name', 'Unknown'),
                        'memory_total': details.get('memory_total', 0) / (1024**3),  # GB
                    }
                except:
                    pass
            return memory_info
    except:
        pass
    return {}

def benchmark_model_inference(model, num_classes, num_iterations=100, warmup=10):
    """
    Benchmark model inference latency and memory.
    
    Args:
        model: Keras model
        num_classes: Number of classes
        num_iterations: Number of inference iterations
        warmup: Number of warmup iterations
    """
    # Create dummy input
    dummy_input = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        _ = model.predict(dummy_input, verbose=0)
    
    # Measure latency
    latencies = []
    for _ in range(num_iterations):
        start = time.time()
        _ = model.predict(dummy_input, verbose=0)
        end = time.time()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    p50_latency = np.percentile(latencies, 50)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    # Measure memory (approximate)
    try:
        if tf.config.list_physical_devices('GPU'):
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            peak_memory_mb = memory_info['peak'] / (1024**2)
        else:
            peak_memory_mb = 0
    except:
        peak_memory_mb = 0
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    return {
        'avg_latency_ms': float(avg_latency),
        'std_latency_ms': float(std_latency),
        'min_latency_ms': float(min_latency),
        'max_latency_ms': float(max_latency),
        'p50_latency_ms': float(p50_latency),
        'p95_latency_ms': float(p95_latency),
        'p99_latency_ms': float(p99_latency),
        'throughput_fps': 1000.0 / avg_latency if avg_latency > 0 else 0,
        'peak_memory_mb': float(peak_memory_mb),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'model_size_mb': total_params * 4 / (1024**2)  # Assuming float32 (4 bytes)
    }

def benchmark_from_file(model_path, num_classes, num_iterations=100):
    """Benchmark a model loaded from file."""
    print(f"Loading model from {model_path}...")
    try:
        model = load_model(model_path, compile=False)
        return benchmark_model_inference(model, num_classes, num_iterations)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def benchmark_from_builder(model_name, num_classes, num_iterations=100):
    """Benchmark a model built from scratch."""
    print(f"Building model {model_name}...")
    try:
        model = build_model(model_name, num_classes, base_trainable=False)
        return benchmark_model_inference(model, num_classes, num_iterations)
    except Exception as e:
        print(f"Error building model: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference performance")
    parser.add_argument("--models", nargs="*", default=None, help="Model names to benchmark")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--iterations", type=int, default=100, help="Number of inference iterations")
    parser.add_argument("--from_files", action="store_true", help="Load models from MODELS_DIR")
    args = parser.parse_args()
    
    models_to_test = args.models or SUPPORTED_MODELS
    results = {}
    
    print("="*60)
    print("Model Inference Benchmarking")
    print("="*60)
    print(f"GPU Info: {get_gpu_memory_info()}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Iterations: {args.iterations}")
    print("="*60)
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")
        
        if args.from_files:
            # Try to find model file
            model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.h5') and model_name.lower() in f.lower()]
            if model_files:
                model_path = os.path.join(MODELS_DIR, model_files[0])
                result = benchmark_from_file(model_path, args.num_classes, args.iterations)
            else:
                print(f"No model file found for {model_name}, building from scratch...")
                result = benchmark_from_builder(model_name, args.num_classes, args.iterations)
        else:
            result = benchmark_from_builder(model_name, args.num_classes, args.iterations)
        
        if result:
            results[model_name] = result
            print(f"\nResults for {model_name}:")
            print(f"  Average Latency: {result['avg_latency_ms']:.2f} ms")
            print(f"  Throughput: {result['throughput_fps']:.2f} FPS")
            print(f"  P95 Latency: {result['p95_latency_ms']:.2f} ms")
            print(f"  Peak Memory: {result['peak_memory_mb']:.2f} MB")
            print(f"  Total Parameters: {result['total_params']:,}")
            print(f"  Model Size: {result['model_size_mb']:.2f} MB")
        else:
            print(f"Failed to benchmark {model_name}")
    
    # Save results
    json_path = os.path.join(RESULTS_DIR, "inference_benchmark.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Saved benchmark results to {json_path}")
    print(f"{'='*60}")
    
    # Print summary table
    if results:
        print("\nSummary Table:")
        print(f"{'Model':<20} {'Latency (ms)':<15} {'Throughput (FPS)':<18} {'Memory (MB)':<15} {'Params':<15}")
        print("-" * 85)
        for model_name, result in sorted(results.items(), key=lambda x: x[1]['avg_latency_ms']):
            print(f"{model_name:<20} {result['avg_latency_ms']:<15.2f} {result['throughput_fps']:<18.2f} "
                  f"{result['peak_memory_mb']:<15.2f} {result['total_params']:<15,}")

if __name__ == "__main__":
    main()

