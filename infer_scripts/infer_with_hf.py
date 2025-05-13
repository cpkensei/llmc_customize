import csv
import os
import time
import numpy as np
import psutil
import torch
import gc
from glob import glob
from threading import Thread
import pynvml
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import uuid

# Set PyTorch environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure model paths and output directory
MODEL_PATHS = ['/root/autodl-tmp/model_save/llama3_8b_sgl/sgl_quant_model_awq_b4']
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

def calculate_params(model):
    """Calculate total number of model parameters."""
    try:
        total_params = sum(p.numel() for p in model.parameters())
        return total_params
    except Exception as e:
        print(f"Debug: Error calculating parameters: {e}")
        return 8e9  # Default to 8B for LLaMA-3 8B

def estimate_flops(params, completion_time, token_count):
    """Estimate FLOPs in Millions based on model parameters and inference time."""
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

def measure_ttft(model, tokenizer, prompt, sampling_params, vram_samples, baseline_vram, device):
    """Measure Time to First Token using streaming."""
    start_time = time.time()
    ttft = 0.0
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        
        # Custom callback to capture first token time
        first_token_received = [False]
        def on_token(token):
            if not first_token_received[0]:
                first_token_received[0] = True
                nonlocal ttft
                ttft = time.time() - start_time
                vram_samples.append(get_vram_usage(baseline_vram))
        
        # Simulate streaming by generating tokens one at a time
        model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            streamer=streamer,
            use_cache=True
        )
        
        if ttft == 0.0:
            print("Debug: No valid text received in stream")
    except Exception as e:
        print(f"Debug: Streaming error: {e}")
        # Fallback to non-streaming
        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=sampling_params["temperature"],
                top_p=sampling_params["top_p"],
                use_cache=True
            )
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
    """Kill residual transformer-related processes to free GPU memory."""
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.pid == current_pid:
                continue
            cmdline = proc.info['cmdline'] or []
            if any('python' in str(arg).lower() and 'transformers' in str(arg).lower() for arg in cmdline):
                print(f"Terminating residual process: PID {proc.pid}, Command: {cmdline}")
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

    all_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_path in MODEL_PATHS:
        model_type = os.path.basename(model_path)
        print(f"\nRunning experiments for model: {model_path} ({model_type})")
        
        baseline_vram = get_vram_usage()
        print(f"Debug: Baseline VRAM before model loading: {baseline_vram:.2f} MB")

        model = None
        tokenizer = None
        for attempt in range(3):
            try:
                print(f"Attempt {attempt + 1} to load model {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
                break
            except Exception as e:
                print(f"Error loading model {model_path} on attempt {attempt + 1}: {e}")
                if attempt < 2:
                    cleanup_processes()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                    time.sleep(2)
                else:
                    print(f"Failed to load model {model_path} after 3 attempts. Skipping.")
                    break
        if model is None or tokenizer is None:
            continue

        total_params = calculate_params(model)
        print(f"Debug: Total parameters: {total_params:,}")

        num_runs = len(prompts)
        valid_results = []

        for prompt in prompts:
            vram_samples = []
            stop_flag = [False]
            vram_thread = Thread(target=monitor_vram, args=(vram_samples, stop_flag, baseline_vram))
            vram_thread.start()

            start_time = time.time()
            ttft = measure_ttft(model, tokenizer, prompt, sampling_params, vram_samples, baseline_vram, device)
            if ttft is None:
                print(f"Error measuring TTFT for prompt '{prompt}': Skipping")
                stop_flag[0] = True
                vram_thread.join()
                continue

            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=sampling_params["temperature"],
                    top_p=sampling_params["top_p"],
                    use_cache=True
                )
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
            token_count = len(tokenizer.encode(output_text)) - len(tokenizer.encode(prompt))
            tokens_per_second = token_count / completion_time if completion_time > 0 else 0
            peak_vram_used_mb = max(vram_samples) if vram_samples else 0.0

            valid_results.append({
                "ttft": ttft,
                "completion_time": completion_time,
                "tokens_per_second": tokens_per_second,
                "vram_peak": peak_vram_used_mb,
                "flops": estimate_flops(total_params, completion_time, token_count)[0],
                "flops_per_second": estimate_flops(total_params, completion_time, token_count)[1]
            })

            print("===============================")
            print(f"Prompt: {prompt}\nGenerated text: {output_text}")
            print(f"TTFT: {ttft:.2f}s, Completion Time: {completion_time:.2f}s, Tokens/s: {tokens_per_second:.2f}")
            print(f"Debug: Peak VRAM for prompt: {peak_vram_used_mb:.2f} MB")

        metrics = calculate_metrics(valid_results)

        results = {
            "model_type": model_type,
            "total_params": total_params,
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
            del model
            del tokenizer
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