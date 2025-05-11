import time
import pynvml
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import matplotlib.pyplot as plt
import numpy as np

# Model and tokenizer setup
model_path = '/root/autodl-tmp/model_save/llama3_8b_new/vllm_quant_model_awq'

if __name__ == '__main__':
    # Initialize NVIDIA management library
    pynvml.nvmlInit()

    # Function to get current GPU memory usage
    def get_gpu_memory(device_index=0):
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem_info.used / 1024**2  # Return memory in MB

    # Lists to store memory usage and timestamps
    memory_usage = []
    timestamps = []
    start_time = time.time()

    # Function to monitor memory in the background
    def monitor_memory(stop_flag, interval=0.1):
        while not stop_flag():
            memory = get_gpu_memory()
            memory_usage.append(memory)
            timestamps.append(time.time() - start_time)
            time.sleep(interval)

    
    model = LLM(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Define prompts and sampling parameters
    prompts = [
        'Hello, my name is',
        'The president of the United States is',
        'The capital of France is',
        'The future of AI is',
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Start memory monitoring in a separate thread
    import threading
    stop_monitoring = False
    monitor_thread = threading.Thread(target=monitor_memory, args=(lambda: stop_monitoring, 0.1))
    monitor_thread.start()

    # Perform model inference
    outputs = model.generate(prompts, sampling_params)

    # Stop memory monitoring
    stop_monitoring = True
    monitor_thread.join()

    # Process and print outputs
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt: {prompt!r}, Generated text: {generated_text!r}')

    # Plot memory usage
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, memory_usage, 'b-', label='GPU Memory Usage (MB)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('GPU Memory Usage Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('gpu_memory_usage.png')

    # Cleanup
    pynvml.nvmlShutdown()