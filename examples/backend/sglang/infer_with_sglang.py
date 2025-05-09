import os
import time
import logging
import numpy as np
import csv
from dataclasses import dataclass
from typing import Optional, Dict, List
import openai
import pynvml

# Configuration
WARMUP_RUNS = 2  # Number of warmup runs to stabilize measurements
GPU_MEMORY_THRESHOLD = 4000  # MB (if VRAM measurement is enabled)
CUDA_VISIBLE_DEVICES = ['0', '1', '2', '3', '4', '5', '6', '7']  # Available GPUs

# Set up logging
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"performance_openai_{timestamp}.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().handlers[0].flush = lambda: logging.getLogger().handlers[0].stream.flush()

# Set up CSV output
def setup_csv(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_file = os.path.join(log_dir, f"performance_results_openai_{timestamp}.csv")
    headers = [
        'model_name', 'model_type', 'algorithm', 'avg_ttft', 'std_ttft',
        'avg_completion_time', 'std_completion_time', 'avg_tokens_per_second',
        'std_tokens_per_second', 'avg_vram_used_mb', 'successful_runs', 'total_runs',
        'flops', 'macs', 'params', 'peak_vram_mb'
    ]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
    return csv_file

def append_to_csv(csv_file, results, model_name, model_type, algo):
    row = [
        model_name,
        model_type,
        algo,
        results.get('avg_ttft', None),
        results.get('std_ttft', None),
        results.get('avg_completion_time', None),
        results.get('std_completion_time', None),
        results.get('avg_tokens_per_second', 0),
        results.get('std_tokens_per_second', 0),
        results.get('avg_vram_used_mb', None),
        results.get('successful_runs', 0),
        results.get('total_runs', 0),
        results.get('model_stats', {}).get('flops', 'N/A'),
        results.get('model_stats', {}).get('macs', 'N/A'),
        results.get('model_stats', {}).get('params', 'N/A'),
        results.get('model_stats', {}).get('peak_vram_mb', None)
    ]
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

@dataclass
class PerformanceMetrics:
    ttft: Optional[float]
    completion_time: Optional[float]
    tokens_per_second: float
    tokens_generated: int
    vram_used_mb: Optional[float]

@dataclass
class ModelStats:
    flops: str
    macs: str
    params: str
    peak_vram_mb: Optional[float]

class OpenAIPerformanceTester:
    def __init__(self, model_name: str, base_url: str, api_key: str, gpu_id: str):
        self.model_name = model_name
        self.client = openai.Client(base_url=base_url, api_key=api_key)
        self.gpu_id = gpu_id
        self.gpu_handle = None
        self.model_type = "openai_api"

        if gpu_id not in CUDA_VISIBLE_DEVICES:
            raise ValueError(f"Invalid GPU ID {gpu_id}. Available GPUs: {CUDA_VISIBLE_DEVICES}")

        try:
            pynvml.nvmlInit()
            physical_gpu_index = CUDA_VISIBLE_DEVICES.index(gpu_id)
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu_index)
        except Exception as e:
            logging.error(f"Error initializing pynvml for GPU {gpu_id}: {str(e)}")
            self.gpu_handle = None

    def _get_vram_usage(self) -> Optional[float]:
        if self.gpu_handle is None:
            return None
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return mem_info.used / 1024 / 1024
        except Exception as e:
            logging.error(f"Error getting VRAM usage for GPU {self.gpu_id}: {str(e)}")
            return None

    def _calculate_model_stats(self) -> ModelStats:
        # Since OpenAI API doesn't expose model internals, these stats are not available
        peak_vram = self._get_vram_usage()
        return ModelStats(flops="N/A", macs="N/A", params="N/A", peak_vram_mb=peak_vram)

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.client:
                self.client = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logging.error(f"Error during cleanup on GPU {self.gpu_id}: {str(e)}")

    def warmup(self, prompt: str, max_tokens: int = 100):
        """Perform warmup runs to stabilize performance measurements."""
        for _ in range(WARMUP_RUNS):
            self.measure_speed(prompt, max_tokens, is_warmup=True)

    def measure_speed(self, prompt: str, max_tokens: int = 100, is_warmup: bool = False) -> PerformanceMetrics:
        first_token_time = None
        tokens_received = 0
        start_time = time.perf_counter()

        try:
            # Since OpenAI API doesn't provide streaming for completions.create, we measure TTFT separately
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=0,
                max_tokens=1  # For TTFT
            )
            first_token_time = time.perf_counter()
            if not response.choices or not response.choices[0].text.strip():
                raise ValueError("No tokens generated for TTFT")

            # Full generation
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=0,
                max_tokens=max_tokens
            )
            end_time = time.perf_counter()

            if response.choices and response.choices[0].text:
                # Approximate token count (OpenAI API doesn't return exact token counts for completions)
                generated_text = response.choices[0].text.strip()
                tokens_received = len(generated_text.split())  # Rough token estimation
            else:
                raise ValueError("No tokens generated")

            completion_time = end_time - start_time
            ttft = first_token_time - start_time if first_token_time else None
            token_generation_time = max(0.0001, end_time - (first_token_time or start_time))
            tokens_per_second = tokens_received / token_generation_time if tokens_received > 0 else 0
            vram_used_mb = self._get_vram_usage()

            if not is_warmup:
                return PerformanceMetrics(
                    ttft=ttft,
                    completion_time=completion_time,
                    tokens_per_second=tokens_per_second,
                    tokens_generated=tokens_received,
                    vram_used_mb=vram_used_mb
                )
            else:
                return PerformanceMetrics(
                    ttft=None,
                    completion_time=None,
                    tokens_per_second=0,
                    tokens_generated=0,
                    vram_used_mb=None
                )

        except Exception as e:
            logging.error(f"Error during generation on GPU {self.gpu_id}: {str(e)}")
            return PerformanceMetrics(
                ttft=None,
                completion_time=None,
                tokens_per_second=0,
                tokens_generated=0,
                vram_used_mb=None
            )

    def measure_vram(self, prompt: str, max_tokens: int = 100) -> Optional[float]:
        peak_vram = self._get_vram_usage()
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=0,
                max_tokens=max_tokens
            )
            current_vram = self._get_vram_usage()
            if current_vram is not None and peak_vram is not None:
                peak_vram = max(peak_vram, current_vram)
            return peak_vram
        except Exception as e:
            logging.error(f"Error during VRAM measurement on GPU {self.gpu_id}: {str(e)}")
            return None

    def run_benchmark(self, prompt: str, max_tokens: int = 100, num_runs: int = 5) -> Dict:
        self.warmup(prompt, max_tokens)

        speed_results: List[PerformanceMetrics] = []
        for _ in range(num_runs):
            metrics = self.measure_speed(prompt, max_tokens)
            speed_results.append(metrics)

        vram_used_mb = self.measure_vram(prompt, max_tokens)
        model_stats = self._calculate_model_stats()

        valid_results = [r for r in speed_results if r.completion_time is not None]
        if not valid_results:
            logging.error(f"All speed runs failed for {self.model_type} model {self.model_name} on GPU {self.gpu_id}")
            return {
                "error": "All speed runs failed",
                "avg_ttft": None,
                "std_ttft": None,
                "avg_completion_time": None,
                "std_completion_time": None,
                "avg_tokens_per_second": 0,
                "std_tokens_per_second": 0,
                "avg_vram_used_mb": vram_used_mb,
                "model_stats": model_stats.__dict__,
                "model_type": self.model_type
            }

        ttfts = [r.ttft for r in valid_results if r.ttft is not None]
        completion_times = [r.completion_time for r in valid_results]
        tokens_per_seconds = [r.tokens_per_second for r in valid_results]

        avg_ttft = np.mean(ttfts) if ttfts else None
        std_ttft = np.std(ttfts) if ttfts else None
        avg_completion_time = np.mean(completion_times)
        std_completion_time = np.std(completion_times)
        avg_tokens_per_second = np.mean(tokens_per_seconds)
        std_tokens_per_second = np.std(tokens_per_seconds)

        results = {
            "avg_ttft": avg_ttft,
            "std_ttft": std_ttft,
            "avg_completion_time": avg_completion_time,
            "std_completion_time": std_completion_time,
            "avg_tokens_per_second": avg_tokens_per_second,
            "std_tokens_per_second": std_tokens_per_second,
            "avg_vram_used_mb": vram_used_mb,
            "successful_runs": len(valid_results),
            "total_runs": num_runs,
            "model_stats": model_stats.__dict__,
            "model_type": self.model_type
        }

        return results

    def __del__(self):
        self.cleanup()
        if self.gpu_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                logging.error(f"Error shutting down pynvml for GPU {self.gpu_id}: {str(e)}")

def run_task(args_dict, gpu_id, log_dir, csv_file):
    tester = None
    try:
        tester = OpenAIPerformanceTester(
            model_name=args_dict['model_name'],
            base_url=args_dict['base_url'],
            api_key=args_dict['api_key'],
            gpu_id=gpu_id
        )
        results = tester.run_benchmark(
            prompt=args_dict['prompt'],
            max_tokens=args_dict['max_tokens'],
            num_runs=args_dict['num_runs']
        )
        append_to_csv(csv_file, results, args_dict['model_name'], args_dict['model_type'], args_dict['algo'])
        logging.info(f"Benchmark completed for {args_dict['model_name']} ({args_dict['model_type']}) on GPU {gpu_id}")
        return results
    except Exception as e:
        logging.error(f"Failed to test {args_dict['model_type']} model {args_dict['model_name']} on GPU {gpu_id}: {str(e)}")
        return None
    finally:
        if tester is not None:
            tester.cleanup()

def main(args_list, log_dir, csv_file):
    gpu_cycle = iter(CUDA_VISIBLE_DEVICES)
    for task in args_list:
        gpu_id = next(gpu_cycle, CUDA_VISIBLE_DEVICES[0])
        logging.info(f"Running task: {task['model_name']} ({task['model_type']}) on GPU {gpu_id}")
        results = run_task(task, gpu_id, log_dir, csv_file)
        if results:
            logging.info(f"Task completed successfully: {task['model_name']}")
        else:
            logging.error(f"Task failed: {task['model_name']}")

if __name__ == "__main__":
    log_dir = "./performance_logs"
    prompt = "Explain quantum computing in simple terms."
    max_tokens = 100
    num_runs = 5

    setup_logging(log_dir)
    csv_file = setup_csv(log_dir)

    # Define the model and API configuration
    args_list = [
        {
            'model_name': 'default',  # Adjust based on your API's model name
            'base_url': 'http://127.0.0.1:30000/v1',
            'api_key': 'EMPTY',
            'model_type': 'openai_api',
            'prompt': prompt,
            'max_tokens': max_tokens,
            'num_runs': num_runs,
            'algo': 'origin'  # Assuming no quantization or sparsity for the API model
        }
    ]

    logging.info(f"Found {len(args_list)} tasks to run.")
    main(args_list=args_list, log_dir=log_dir, csv_file=csv_file)