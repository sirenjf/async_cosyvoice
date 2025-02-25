# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import asyncio
from concurrent import futures
import argparse
from typing import AsyncGenerator, Callable, Tuple, AsyncIterator

import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import grpc
from grpc import aio

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/../..')
sys.path.append(f'{ROOT_DIR}/../../third_party/Matcha-TTS')
from async_cosyvoice.async_cosyvoice import AsyncCosyVoice2
from async_cosyvoice.async_grpc.utils import convert_audio_tensor_to_bytes, convert_audio_bytes_to_tensor

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


class CosyVoiceServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    def __init__(self, args):
        try:
            self.cosyvoice = AsyncCosyVoice2(args.model_dir)
        except Exception:
            raise RuntimeError('no valid model_type! just support AsyncCosyVoice2.')
        logging.info('grpc service initialized')

    async def Inference(self, request: cosyvoice_pb2.Request, context: aio.ServicerContext) -> AsyncIterator[
        cosyvoice_pb2.Response]:
        """统一异步流式处理入口"""
        try:
            # 获取处理器和预处理后的参数
            processor, processor_args = await self._prepare_processor(request)

            # 通过通用处理器生成响应
            async for response in self._handle_generic(processor, processor_args):
                yield response

        except Exception as e:
            logging.error(f"Request processing failed: {str(e)}", exc_info=True)
            await context.abort(
                code=grpc.StatusCode.INTERNAL,
                details=f"Processing error: {str(e)}"
            )

    async def _prepare_processor(self, request: cosyvoice_pb2.Request) -> Tuple[Callable, list]:
        """预处理并返回处理器及其参数"""
        print(request)
        match request.WhichOneof('request_type'):
            case 'sft_request':
                return self.cosyvoice.inference_sft, [
                    request.sft_request.tts_text,
                    request.sft_request.spk_id,
                    request.stream,
                    request.speed,
                    request.text_frontend
                ]
            case 'zero_shot_request':
                prompt_audio = await asyncio.to_thread(
                    convert_audio_bytes_to_tensor,
                    request.zero_shot_request.prompt_audio
                )
                return self.cosyvoice.inference_zero_shot, [
                    request.zero_shot_request.tts_text,
                    request.zero_shot_request.prompt_text,
                    prompt_audio,
                    request.stream,
                    request.speed,
                    request.text_frontend
                ]
            case 'cross_lingual_request':
                prompt_audio = await asyncio.to_thread(
                    convert_audio_bytes_to_tensor,
                    request.cross_lingual_request.prompt_audio
                )
                return self.cosyvoice.inference_cross_lingual, [
                    request.cross_lingual_request.tts_text,
                    prompt_audio,
                    request.stream,
                    request.speed,
                    request.text_frontend
                ]
            case 'instruct2_request':
                prompt_audio = await asyncio.to_thread(
                    convert_audio_bytes_to_tensor,
                    request.instruct2_request.prompt_audio
                )
                return self.cosyvoice.inference_instruct2, [
                    request.instruct2_request.tts_text,
                    request.instruct2_request.instruct_text,
                    prompt_audio,
                    request.stream,
                    request.speed,
                    request.text_frontend
                ]
            case 'instruct2_by_spk_id_request':
                return self.cosyvoice.inference_instruct2_by_spk_id, [
                    request.instruct2_by_spk_id_request.tts_text,
                    request.instruct2_by_spk_id_request.instruct_text,
                    request.instruct2_by_spk_id_request.spk_id,
                    request.stream,
                    request.speed,
                    request.text_frontend
                ]
            case 'zero_shot_by_spk_id_request':
                return self.cosyvoice.inference_zero_shot_by_spk_id, [
                    request.zero_shot_by_spk_id_request.tts_text,
                    request.zero_shot_by_spk_id_request.spk_id,
                    request.stream,
                    request.speed,
                    request.text_frontend
                ]
            case _:
                raise ValueError("Invalid request type")

    async def _handle_generic(
            self,
            processor: Callable,
            processor_args: list
    ) -> AsyncGenerator[cosyvoice_pb2.Response, None]:
        """通用流式处理管道"""
        logging.debug(f"Processing with {processor.__name__}")
        async for model_chunk in processor(*processor_args):
            # 将同步数据转换操作放入线程池
            audio_bytes = await asyncio.to_thread(
                convert_audio_tensor_to_bytes,
                model_chunk['tts_speech']
            )
            yield cosyvoice_pb2.Response(tts_audio=audio_bytes)

async def serve(args):
    server = aio.server(
        migration_thread_pool=futures.ThreadPoolExecutor(max_workers=args.max_conc),
        maximum_concurrent_rpcs=args.max_conc
    )
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(CosyVoiceServiceImpl(args), server)
    server.add_insecure_port(f'0.0.0.0:{args.port}')
    await server.start()
    logging.info(f"Server listening on 0.0.0.0:{args.port}")
    await server.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--max_conc', type=int, default=4)
    parser.add_argument('--model_dir', type=str,
                        default='../../pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()

    # 启动异步事件循环
    asyncio.run(serve(args))