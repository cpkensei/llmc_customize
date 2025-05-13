import csv
import os
import time
import numpy as np
import sglang as sgl
from sglang.srt.conversation import chat_templates
import psutil
import torch
import gc
from glob import glob
from threading import Thread
import pynvml
import uuid
from transformers import AutoTokenizer

# Set PyTorch environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure model paths and output directory
MODEL_PATHS = glob('/root/autodl-tmp/model_save/llama3_all/sgl*')
MODEL_PATHS.extend(glob('/root/autodl-tmp/model_save/llama3_all/fake*'))
LOG_DIR = "/root/autodl-tmp/performance_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_vram_usage(baseline_vram=0.0):
    """Get VRAM usage in MB for the current GPU using pynvml, relative to baseline."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_used = mem_info.used / 1024**2
        vram_delta = max(0.0, vram_used - baseline_vram)
        print(f"Debug: VRAM used (pynvml): {vram_used:.2f} MB, Delta: {vram_delta:.2f} MB")
        pynvml.nvmlShutdown()
        return vram_delta
    except Exception as e:
        print(f"Debug: Error getting VRAM with pynvml: {e}")
        return 0.0

def calculate_params(model, model_path):
    """Calculate parameter memory size in bytes based on actual parameter data types."""
    try:
        total_bytes = 0
        total_params = 0
        
        # Determine quantization type from model path for AWQ-specific handling
        model_path_lower = model_path.lower()
        is_awq = 'awq' in model_path_lower
        if 'b4' in model_path_lower or 'w4' in model_path_lower:
            quant_bits = 4  # 4-bit weights for AWQ
        elif 'w8' in model_path_lower:
            quant_bits = 8  # 8-bit weights for AWQ
        else:
            quant_bits = None  # No quantization, assume float16
        
        for name, param in model.named_parameters():
            num_elements = param.numel()
            total_params += num_elements
            dtype = param.dtype
            
            # Map dtype to bits per parameter
            if dtype == torch.float16:
                bits_per_param = 16
            elif dtype == torch.float32:
                bits_per_param = 32
            elif dtype == torch.int8:
                if is_awq and quant_bits == 4:
                    # AWQ 4-bit weights are packed into int8 tensors (2 weights per byte)
                    bits_per_param = 4
                else:
                    bits_per_param = 8
            else:
                # Default to 16 bits for unknown dtypes
                print(f"Debug: Unknown dtype {dtype} for parameter {name}, assuming 16 bits")
                bits_per_param = 16
            
            # Calculate bytes for this parameter
            bytes_per_param = bits_per_param / 8
            param_bytes = num_elements * bytes_per_param
            total_bytes += param_bytes
            print(f"Debug: Param {name}: {num_elements:,} elements, dtype={dtype}, {bits_per_param} bits, {param_bytes:,} bytes")
        
        print(f"Debug: Total parameters: {total_params:,}, Total bytes: {total_bytes:,}")
        return total_bytes
    except Exception as e:
        print(f"Debug: Error calculating parameters: {e}")
        # Default to 8B parameters in float16 (8e9 * 2 bytes)
        return 8e9 * 2

def estimate_flops(param_bytes, completion_time, token_count):
    """Estimate FLOPs in Millions based on parameter memory size and inference time."""
    # Convert bytes to number of parameters (assuming float16 for unquantized params for FLOPs)
    params = param_bytes / 2  # 2 bytes per param for float16 equivalent
    flops = 2 * params * token_count / 1e6  # Convert to Millions
    flops_per_second = flops / completion_time if completion_time > 0 else 0
    return flops, flops_per_second

def calculate_metrics(results):
    """Calculate average and standard deviation for metrics."""
    ttfts = [r["ttft"] for r in results]
    completion_times = [r["completion_time"] for r in results]
    tokens_per_sec = [r["tokens_per_second"] for r in results]
    vram_peaks = [r["vram_peak"] for r in results]
    flops = [r["flops"] for r in results]
    flops_per_sec = [r["flops_per_second"] for r in results]
    
    return {
        "avg_ttft": np.mean(ttfts) if ttfts else 0.0,
        "std_ttft": np.std(ttfts) if ttfts else 0.0,
        "avg_completion_time": np.mean(completion_times) if completion_times else 0.0,
        "std_completion_time": np.std(completion_times) if completion_times else 0.0,
        "avg_tokens_per_second": np.mean(tokens_per_sec) if tokens_per_sec else 0.0,
        "std_tokens_per_second": np.std(tokens_per_sec) if tokens_per_sec else 0.0,
        "peak_vram_used_mb": max(vram_peaks) if vram_peaks else 0.0,
        "avg_flops_million": np.mean(flops) if flops else 0.0,
        "std_flops_million": np.std(flops) if flops else 0.0,
        "avg_flops_per_second_million": np.mean(flops_per_sec) if flops_per_sec else 0.0,
        "std_flops_per_second_million": np.std(flops_per_sec) if flops_per_sec else 0.0
    }

def measure_ttft(llm, prompt, sampling_params, vram_samples, baseline_vram):
    """Measure Time to First Token using streaming and collect VRAM samples."""
    start_time = time.time()
    ttft = 0.0
    
    try:
        stream_iter = llm.generate([prompt], sampling_params, stream=True)
        for chunk in stream_iter:
            print(f"Debug: Stream chunk: {chunk}")
            vram_samples.append(get_vram_usage(baseline_vram))
            if 'text' in chunk and chunk['text']:
                ttft = time.time() - start_time
                break
        if ttft == 0.0:
            print("Debug: No valid text received in stream")
    except Exception as e:
        print(f"Debug: Streaming error: {e}")
        try:
            output = llm.generate([prompt], sampling_params)[0]
            print(f"Debug: Non-stream output: {output}")
            vram_samples.append(get_vram_usage(baseline_vram))
            ttft = time.time() - start_time
        except Exception as e:
            print(f"Debug: Non-streaming error: {e}")
            return None
    
    return ttft

def monitor_vram(vram_samples, stop_flag, baseline_vram):
    """Continuously monitor VRAM usage in a separate thread."""
    while not stop_flag[0]:
        vram_samples.append(get_vram_usage(baseline_vram))
        time.sleep(0.01)

def cleanup_processes():
    """Kill residual SGLang-related processes to free GPU memory."""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.pid == current_pid:
                continue
            cmdline = proc.info['cmdline'] or []
            if any('sglang' in str(arg).lower() for arg in cmdline):
                print(f"Terminating residual SGLang process: PID {proc.pid}, Command: {cmdline}")
                proc.terminate()
                proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
            print(f"Error terminating process {proc.pid}: {e}")
            continue

if __name__ == "__main__":
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95
    }

    try:
        default_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    except Exception as e:
        print(f"Debug: Error loading default tokenizer: {e}")
        default_tokenizer = None

    all_results = []

    for model_path in MODEL_PATHS:
        model_type = os.path.basename(model_path)
        print(f"\nRunning experiments for model: {model_path} ({model_type})")
        
        baseline_vram = get_vram_usage()
        print(f"Debug: Baseline VRAM before model loading: {baseline_vram:.2f} MB")

        llm = None
        for attempt in range(3):
            try:
                print(f"Attempt {attempt + 1} to initialize model {model_path}")
                llm = sgl.Engine(model_path=model_path)
                break
            except Exception as e:
                print(f"Error initializing model {model_path} on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    cleanup_processes()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    time.sleep(2)
                else:
                    print(f"Failed to initialize model {model_path} after 3 attempts. Skipping.")
                    break
        if llm is None:
            continue

        model = None
        tokenizer = default_tokenizer
        param_bytes = 8e9 * 2  # Default to 8B parameters in float16
        try:
            model = llm.model_runner.model if hasattr(llm, 'model_runner') and hasattr(llm.model_runner, 'model') else None
            tokenizer = llm.tokenizer if hasattr(llm, 'tokenizer') else default_tokenizer
            if model:
                param_bytes = calculate_params(model, model_path)
            if model is None or tokenizer is None:
                print("Debug: Could not access model or tokenizer from SGLang engine. Using default tokenizer.")
        except Exception as e:
            print(f"Debug: Error accessing model/tokenizer: {e}")

        num_runs = len(prompts)
        valid_results = []

        for prompt in prompts:
            vram_samples = []
            stop_flag = [False]
            vram_thread = Thread(target=monitor_vram, args=(vram_samples, stop_flag, baseline_vram))
            vram_thread.start()

            start_time = time.time()
            ttft = measure_ttft(llm, prompt, sampling_params, vram_samples, baseline_vram)
            if ttft is None:
                print(f"Error measuring TTFT for prompt '{prompt}': Skipping")
                stop_flag[0] = True
                vram_thread.join()
                continue

            try:
                output = llm.generate([prompt], sampling_params)[0]
                output_text = output['text']
                vram_samples.append(get_vram_usage(baseline_vram))
            except Exception as e:
                print(f"Error generating output for prompt '{prompt}': {e}")
                stop_flag[0] = True
                vram_thread.join()
                continue
            end_time = time.time()
            vram_samples.append(get_vram_usage(baseline_vram))

            stop_flag[0] = True
            vram_thread.join()

            completion_time = end_time - start_time
            # Use tokenizer for precise token counting if available
            if tokenizer:
                token_count = len(tokenizer.encode(output_text)) - len(tokenizer.encode(prompt))
            else:
                # Fallback to whitespace splitting
                token_count = len(output_text.split())
            tokens_per_second = token_count / completion_time if completion_time > 0 else 0
            peak_vram_used_mb = max(vram_samples) if vram_samples else 0.0

            valid_results.append({
                "ttft": ttft,
                "completion_time": completion_time,
                "tokens_per_second": tokens_per_second,
                "vram_peak": peak_vram_used_mb,
                "flops": estimate_flops(param_bytes, completion_time, token_count)[0],
                "flops_per_second": estimate_flops(param_bytes, completion_time, token_count)[1]
            })

            print("===============================")
            print(f"Prompt: {prompt}\nGenerated text: {output_text}")
            print(f"TTFT: {ttft:.2f}s, Completion Time: {completion_time:.2f}s, Tokens/s: {tokens_per_second:.2f}")
            print(f"Debug: Peak VRAM for prompt: {peak_vram_used_mb:.2f} MB")

        metrics = calculate_metrics(valid_results)

        results = {
            "model_type": model_type,
            "param_bytes": param_bytes,
            "avg_ttft": metrics["avg_ttft"],
            "std_ttft": metrics["std_ttft"],
            "avg_completion_time": metrics["avg_completion_time"],
            "std_completion_time": metrics["std_completion_time"],
            "avg_tokens_per_second": metrics["avg_tokens_per_second"],
            "std_tokens_per_second": metrics["std_tokens_per_second"],
            "peak_vram_used_mb": metrics["peak_vram_used_mb"],
            "avg_flops_million": metrics["avg_flops_million"],
            "std_flops_million": metrics["std_flops_million"],
            "avg_flops_per_second_million": metrics["avg_flops_per_second_million"],
            "std_flops_per_second_million": metrics["std_flops_per_second_million"],
            "successful_runs": len(valid_results),
            "total_runs": num_runs
        }

        all_results.append(results)

        try:
            del llm
            llm = None
            cleanup_processes()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            print(f"GPU memory and processes cleared for {model_path}")
        except Exception as e:
            print(f"Error clearing GPU memory or processes for {model_path}: {e}")
        
        print(f"Completed experiments for {model_path}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(LOG_DIR, f"performance_batch_{timestamp}.csv")
    with open(csv_file, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    print(f"All performance metrics written to {csv_file}")