<div align="center">

# Async CosyVoice
</div>

## 项目说明

该项目是为了加速cosyvoice2的推理，仅支持linux系统，依赖vllm。
1. **使用vllm加速llm部分的推理**
2. flow部分的推理**未优化**，流式或并发时会隐形llm部分的推理，需大佬优化
3. grpc-async服务端

---
### 环境准备(使用python=3.10.16)
```bash
conda create -n cosyvoice2 python=3.10.16
conda activate cosyvoice2
```

---
### 使用方式

1. 先clone cosyvoice项目
````
git clone https://github.com/FunAudioLLM/CosyVoice.git
````

2. 在cosyvoice项目路径下再clone本项目
```bash
cd CosyVoice
git clone https://github.com/qi-hua/async_cosyvoice.git
```

3. 在async_cosyvoice目录下安装所有依赖
```bash
cd async_cosyvoice
uv pip install -r requirements.txt
#或者使用 pip install -r requirements.txt
```

4. 下载CosyVoice2-0.5B模型文件 

    - 下载地址：https://www.modelscope.cn/models/iic/CosyVoice2-0.5B/
    - 保存位置为：CosyVoice/pretrained_models
    - 如手动下载则不用下载CosyVoice-BlankEN文件夹下的文件，已经将其中需要的文件修改并复制到CosyVoice2-0.5B文件夹下了；

5. 将 async_cosyvoice/CosyVoice2-0.5B 文件夹下的文件复制到下载的CosyVoice2-0.5B模型文件夹下
```bash
cp async_cosyvoice/CosyVoice2-0.5B/* pretrained_models/CosyVoice2-0.5B/
```

6. 在config.py文件中设置vllm的AsyncEngineArgs参数、SamplingParams参数
```python
ENGINE_ARGS = {
    "gpu_memory_utilization": 0.4,
    "max_num_batched_tokens": 1024,
    "max_model_len": 1024,
    "max_num_seqs": 256,
}

SAMPLING_PARAMS = {
    "max_tokens": 1024,
}
```

7. 启动grpc服务
```bash
#1.由 proto 文件生成依赖代码
cd grpc-async
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto
python server.py --model_path=/path/to/model
```

8. 测试grpc服务
```bash
python client.py
```

---
### 说明：

1. 目前使用的 vllm 最新版本，并开启了 VLLM_USE_V1 = '1'
2. 目前cuda环境是12.4, 部分依赖文件使用的较新版本 vllm==0.7.3 torch==2.5.1 onnxruntime-gpu==1.19.0
3. 如果使用load_trt，需手动安装依赖tensorrt
4. 如在其他地方使用，需要参照
```python
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/../..')
sys.path.append(f'{ROOT_DIR}/../../third_party/Matcha-TTS')
```