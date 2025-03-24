import os
import sys
import asyncio
import time
import numpy as np
import soundfile as sf
import librosa
import wave

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/../../..')
sys.path.append(f'{ROOT_DIR}/../../../third_party/Matcha-TTS')
import logging
import argparse
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import grpc
from grpc import aio

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def load_wav(wav_path: str, target_sr: int) -> np.ndarray:
    """使用 soundfile 替代 torchaudio 加载音频"""
    try:
        data, sample_rate = sf.read(wav_path, dtype='float32')
        # 多声道转单声道
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        # 重采样
        if sample_rate != target_sr:
            data = librosa.resample(
                data,
                orig_sr=sample_rate,
                target_sr=target_sr
            )
        # 添加 batch 维度 (1, samples)
        return data.reshape(1, -1)
    except Exception as e:
        logging.error(f"Error loading {wav_path}: {str(e)}")
        raise

def convert_audio_bytes_to_ndarray(raw_audio: bytes) -> np.ndarray:
    """字节转 numpy 数组"""
    return np.frombuffer(raw_audio, dtype=np.float32).reshape(1, -1)

def convert_audio_ndarray_to_bytes(array: np.ndarray) -> bytes:
    """numpy 数组转字节"""
    return array.astype(np.float32).tobytes()

def construct_request(args, tts_text):
    request = cosyvoice_pb2.Request()
    request.stream = args.stream
    request.speed = args.speed
    request.text_frontend = args.text_frontend

    request.format = args.format

    # 构造请求
    if args.mode == 'sft':
        logging.info('Constructing sft request')
        sft_request = request.sft_request
        sft_request.tts_text = tts_text
        sft_request.spk_id = args.spk_id
    elif args.mode == 'zero_shot':
        logging.info('Constructing zero_shot request')
        zero_shot_request = request.zero_shot_request
        zero_shot_request.tts_text = tts_text
        zero_shot_request.prompt_text = args.prompt_text
        prompt_speech = load_wav(args.prompt_wav, 16000)
        zero_shot_request.prompt_audio = convert_audio_ndarray_to_bytes(prompt_speech)
    elif args.mode == 'cross_lingual':
        logging.info('Constructing cross_lingual request')
        cross_lingual_request = request.cross_lingual_request
        cross_lingual_request.tts_text = tts_text
        prompt_speech = load_wav(args.prompt_wav, 16000)
        cross_lingual_request.prompt_audio = convert_audio_ndarray_to_bytes(prompt_speech)
    elif args.mode == 'instruct2':
        logging.info('Constructing instruct request')
        instruct2_request = request.instruct2_request
        instruct2_request.tts_text = tts_text
        instruct2_request.instruct_text = args.instruct_text
        prompt_speech = load_wav(args.prompt_wav, 16000)
        instruct2_request.prompt_audio = convert_audio_ndarray_to_bytes(prompt_speech)
    elif args.mode == 'instruct2_by_spk_id':
        logging.info('Constructing instruct2_by_spk_id request')
        instruct2_by_spk_id_request = request.instruct2_by_spk_id_request
        instruct2_by_spk_id_request.tts_text = tts_text
        instruct2_by_spk_id_request.instruct_text = args.instruct_text
        instruct2_by_spk_id_request.spk_id = args.spk_id
    else:
        logging.info('Constructing zero_shot_by_spk_id request')
        zero_shot_by_spk_id_request = request.zero_shot_by_spk_id_request
        zero_shot_by_spk_id_request.tts_text = tts_text
        zero_shot_by_spk_id_request.spk_id = args.spk_id
    return request


async def main(args):
    async with aio.insecure_channel(f"{args.host}:{args.port}") as channel:
        stub = cosyvoice_pb2_grpc.CosyVoiceStub(channel)

        # 异步流式接收响应
        try:
            start_time = time.time()
            last_time = start_time
            chunk_index = 0
            tts_audio = b''
            if args.stream_input:
                
                def text_generator(args):
                    text_list = ["不是因为你的每一个人生选择都是错的，", "而是你根本不知道怎么做才算对得起一次选择。"]
                    yield construct_request(args, "")
                    for segment in text_list:
                        yield construct_request(args, segment)

                response_stream = stub.StreamInference(text_generator(args))
            else:
                request = construct_request(args, args.tts_text)
                response_stream = stub.Inference(request)

            async for response in response_stream:
                if response.tts_audio:
                    tts_audio += response.tts_audio
                    logging.info(f'Received audio chunk {len(response.tts_audio)} bytes, chunk index: {chunk_index},' +
                                 f'cost {time.time() - last_time:.3f}s  all cost {time.time() - start_time:.3f}s ')
                    last_time = time.time()
                    chunk_index += 1

            # 音频后处理
            if tts_audio:

                audio_data = np.frombuffer(tts_audio, dtype=np.int16)

                # 保存为 WAV 文件
                with wave.open(args.tts_wav, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # 单声道
                    wav_file.setsampwidth(2)  # 16位
                    wav_file.setframerate(args.target_sr)
                    wav_file.writeframes(audio_data.tobytes())
                    print("sample rate: ", args.target_sr)

                duration = 0
                logging.info(f'Audio saved to {args.tts_wav} (duration: {duration:.2f}s) ' +
                             f'cost {time.time() - last_time:.3f}s  all cost {time.time() - start_time:.3f}s ')
            else:
                logging.warning('Received empty audio response')

        except grpc.RpcError as e:
            logging.error(f"RPC error occurred: {e.code()}: {e.details()}")
        except Exception as e:
            logging.error(f"Processing failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream_input', action="store_true", help="测试流式输入，用于双工流式")
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--mode', default='zero_shot_by_spk_id',
                        choices=['sft', 'zero_shot', 'cross_lingual', 'instruct2',
                                 'instruct2_by_spk_id', 'zero_shot_by_spk_id'],
                        help='Request mode')
    parser.add_argument('--stream', action='store_true',
                        help='Streaming inference mode')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speed up the audio')
    parser.add_argument('--text_frontend', type=bool, default=True,
                        help='Text frontend mode')
    parser.add_argument('--tts_text', type=str,
                        default="不是因为你的每一个人生选择都是错的，而是你根本不知道怎么做才算对得起一次选择。")
    parser.add_argument('--spk_id', type=str, default='001')
    parser.add_argument('--prompt_text', type=str,
                        default='希望你以后能够做的比我还好呦。')
    parser.add_argument('--prompt_wav', type=str,
                        default='../../../asset/zero_shot_prompt.wav')
    parser.add_argument('--format', type=str,
                        default='pcm')
    parser.add_argument('--instruct_text', type=str,
                        default='Theo \'Crimson\', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.')
    parser.add_argument('--tts_wav', type=str, default='demo.wav')
    parser.add_argument('--target_sr', type=int, default=24000,
                        help='Target sample rate for output audio, cosyvoice2 is 24000')
    args = parser.parse_args()

    # 运行异步主函数
    asyncio.run(main(args))

    # python client.py --format "" --stream --stream_input