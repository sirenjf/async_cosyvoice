<div align="center">

# Async CosyVoice
</div>

## 项目说明

该项目是为了加速cosyvoice2的推理，仅支持linux系统，依赖vllm。
1. **使用vllm加速llm部分的推理**
2. flow部分的推理**未优化**，无法直接批处理，建议使用官方的 load_jit load_trt 模式
3. 单任务流式情况下，首包延迟在250-350ms左右，但耗时较非流式有所增加
4. grpc-async服务端

---
### 环境准备(使用python=3.10.16)
```bash
conda create -n cosyvoice2 python=3.10.16 -y
conda activate cosyvoice2
```

---
### 使用方式

1. 先clone cosyvoice项目
````
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
git submodule update --init --recursive

# 安装系统依赖
conda install -y -c conda-forge pynini==2.1.5
sudo apt-get update
sudo apt-get install sox libsox-dev -y
````

2. 在cosyvoice项目路径下再clone本项目，或添加本项目为子模块
```bash
git clone https://github.com/qi-hua/async_cosyvoice.git
```

3. 在async_cosyvoice目录下安装所有依赖
```bash
cd async_cosyvoice
pip install -r requirements.txt
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
    # 根据实际情况设置 
    "gpu_memory_utilization": 0.4,
    "max_num_batched_tokens": 1024,
    "max_model_len": 2048,
    "max_num_seqs": 256,
}

SAMPLING_PARAMS = {
    "max_tokens": 2048,
}
```

7. 启动grpc服务
```bash
#1.由 proto 文件生成依赖代码
cd runtime/async_grpc
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. cosyvoice.proto
python server.py
```

8. 测试grpc服务
```bash
python client.py
```

---
### 测试效果

- 测试代码: [speed_test.ipynb](speed_test.ipynb)
- 测试环境: Intel i5-12400 CPU, 48GB RAM, 1x NVIDIA GeForce RTX 4070
- 运行环境: Ubuntu 24.04.1 LTS, cuda 12.4, python 3.10.16
- 测试说明: tts_text为中文，使用 **jit 和 trt**
- 测试结果: 具体数据及过程详见[speed_test.ipynb](speed_test.ipynb)
  - 进行非流式推理时，运行单任务推理rtf能从原来的0.25-0.30，经过vllm加速后能达到0.1-0.15；
  - 加速后进行流式推理，首包延迟能在250-350ms之间；
  - 流式推理会降低速度，建议使用 inference_zero_shot_by_spk_id 的非流式推理；
  - 并发推理时，建议使用 load_jit 或 load_trt，加快 flow 部分的推理；
  - 非流式并发能够有效支持20个并发（rtf<1），流式并发仅能支持5个并发；

---
### 说明：

1. 目前使用的 vllm 最新版本，并开启了 VLLM_USE_V1 = '1'
2. 目前cuda环境是12.4, 部分依赖文件使用的较新版本 vllm==0.7.3 torch==2.5.1 onnxruntime-gpu==1.19.0
3. 如果使用 load_trt，能有效提升flow模型的推理速度，但显存占用较大(默认启动会占用 4.7G 显存，应该可以设置减少)。
4. 启动 cosyvoice2 实例后，需要进行**预热**，推理10次以上，加载frontend模型以及 预热trt模型。
5. 如在其他地方使用，需要参照async_grpc中的server.py设置sys.path，以正确的引用 cosyvoice、async_cosyvoice
```python
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/../../..')
sys.path.append(f'{ROOT_DIR}/../../../third_party/Matcha-TTS')
```
6. 第一次运行时，如果使用的WeTextProcessing则先运行下面的代码，生成缓存，以免frontend过滤掉儿化音。代码frontend中overwrite_cache=False避免了后续运行时重复生成。
```python
from tn.chinese.normalizer import Normalizer as ZhNormalizer
zh_tn_model = ZhNormalizer(overwrite_cache=True)
```

---
### spk2info说明

spk2info.pt文件中保存了 prompt_text及其token、embedding数据，可以用于 sft、inference_zero_shot_by_spk_id、inference_instruct2_by_spk_id，不用传递参考prompt_text、及音频数据，直接传递spk_id即可。

通过 frontend.**generate_spk_info** 函数生成新的 spk2info，需要传入的参数为：

spk_id: str, prompt_text: str, prompt_speech_16k: torch.Tensor, resample_rate:int=24000, name: str=None。
