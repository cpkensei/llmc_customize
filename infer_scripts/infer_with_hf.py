import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pynvml
import time

# 初始化 pynvml
def init_pynvml():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)  # 假设使用 GPU 0

# 监测显存
def get_peak_memory(handle):
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**2  # 转换为 MB

# 指定模型目录
model_path = "/root/autodl-tmp/model_save/llama3_8b_sgl/sgl_quant_model_shortGPT_9"

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 验证模型加载
print(model)

# 初始化 pynvml
handle = init_pynvml()
peak_memory = 0

# 推理
prompt = "the capital city of China is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 开始显存监测
start_time = time.time()
with torch.no_grad():  # 禁用梯度计算以节省显存
    outputs = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )
    # 记录推理过程中的峰值显存
    current_memory = get_peak_memory(handle)
    peak_memory = max(peak_memory, current_memory)

# 解码输出
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")

# 结束时间和显存统计
end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")
print(f"Peak memory usage: {peak_memory:.2f} MB")

# 清理 pynvml
pynvml.nvmlShutdown()