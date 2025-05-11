import sys
autoawq_path = '/root/autodl-tmp/AutoAWQ'
sys.path.append(autoawq_path)
import torch
import pynvml
import time
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

# 初始化 pynvml
def init_pynvml():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设使用 GPU 0

# 获取当前显存占用
def get_peak_memory(handle):
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**2  # 转换为 MB

# 添加 AutoAWQ 路径


# 模型和分词器路径
model_path = '/root/autodl-tmp/model_save/llama3_8b_new/autoawq_quant_model_awq'

# 加载分词器和流式输出
tokenizer = AutoTokenizer.from_pretrained(model_path)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 加载量化模型
model = AutoAWQForCausalLM.from_quantized(
    model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
)

# 初始化 pynvml
handle = init_pynvml()
peak_memory = 0

# 推理
prompt_text = 'The president of the United States is '
inputs = tokenizer(prompt_text, return_tensors='pt').to('cuda')

# 开始显存监测和推理
start_time = time.time()
with torch.no_grad():  # 禁用梯度计算以节省显存
    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=100,
        streamer=streamer,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    )
    # 记录推理过程中的峰值显存
    current_memory = get_peak_memory(handle)
    peak_memory = max(peak_memory, current_memory)

# 打印推理时间和峰值显存
end_time = time.time()
print(f"\nInference time: {end_time - start_time:.2f} seconds")
print(f"Peak memory usage: {peak_memory:.2f} MB")

# 清理 pynvml
pynvml.nvmlShutdown()