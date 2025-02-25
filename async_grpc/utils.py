import torch
import numpy as np

# def convert_audio_bytes_to_tensor(self, raw_audio: bytes) -> torch.Tensor:
#     """同步音频转换方法"""
#     return (
#             torch.from_numpy(np.array(np.frombuffer(raw_audio, dtype=np.int16)))
#             .unsqueeze(0).float() / (2 ** 15)
#     )
#
# def convert_audio_tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
#     """同步Tensor转字节方法"""
#     return (tensor.numpy() * (2 ** 15)).astype(np.int16).tobytes()

# --------------------------
# 移除音频数据的转换，直接使用传递 float32 数据
# --------------------------
def convert_audio_bytes_to_tensor(raw_audio: bytes) -> torch.Tensor:
    """同步音频转换方法"""
    return torch.from_numpy(np.array(np.frombuffer(raw_audio, dtype=np.float32))).unsqueeze(0)

def convert_audio_tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """同步Tensor转字节方法"""
    return tensor.numpy().astype(np.float32).tobytes()