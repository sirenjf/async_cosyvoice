# vllm settings
# from vllm.engine.arg_utils import AsyncEngineArgs
# AsyncEngineArgs
ENGINE_ARGS = {
    "block_size": 16,
    "swap_space": 0,
    # "enforce_eager": True,
    "gpu_memory_utilization": 0.4,
    "max_num_batched_tokens": 1024,
    "max_model_len": 1024,
    "max_num_seqs": 256,
    "disable_log_requests": True,
    "disable_log_stats": True,
}

from vllm.sampling_params import RequestOutputKind
# SamplingParams
SAMPLING_PARAMS = {
    "temperature": 1,  # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
    "top_p": 1,       # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
    "top_k": 25,
    # "min_tokens": 80,       # 不支持设置最小的tokens数量设置，开启后vllm直接崩溃，无法启动
    # "presence_penalty": 1.0,    # 不支持设置
    # "frequency_penalty": 0.0,   # 不支持设置
    "max_tokens": 1024,
    "detokenize": False,
    "ignore_eos": False,
    "output_kind": RequestOutputKind.DELTA  # 设置为DELTA，如调整该参数，请同时调整llm_inference的处理代码
}
