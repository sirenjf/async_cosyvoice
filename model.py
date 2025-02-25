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
import asyncio
import logging
import os
from typing import Generator, List
import torch
import numpy as np
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid

from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
from cosyvoice.hifigan.generator import HiFTGenerator
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt

# 启用vllm V1版本
os.environ["VLLM_USE_V1"] = '1'
from vllm import  AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm import ModelRegistry
from async_cosyvoice.vllm_use_cosyvoice2_model import CosyVoice2Model as CosyVoice2LLM
ModelRegistry.register_model("CosyVoice2Model", CosyVoice2LLM)



def tensor_to_list(tensor: torch.tensor):
    return tensor.view(-1).cpu().numpy().tolist()

class CosyVoice2Model:

    def __init__(self,
         model_dir: str,
         flow: CausalMaskedDiffWithXvec | torch.nn.Module,
         hift: HiFTGenerator | torch.nn.Module,
         fp16: bool,
         mix_ratio: List[int] = None,
    ):
        # model_dir = "/home/qihua/github/FunAudioLLM-APP/cosyvoice/pretrained_models/CosyVoice2-0.5B"
        # vllm engine 的参数配置
        # TODO： 配置文件写入新的 cosyvoice.yaml 文件中
        engine_args = AsyncEngineArgs(
            model=model_dir,
            block_size=16,
            swap_space=0,
            # enforce_eager=True,
            gpu_memory_utilization=0.4,
            max_num_batched_tokens=1024,
            # max_model_len=4096,
            max_model_len=1024,
            max_num_seqs=256,
            disable_log_requests=True,
            disable_log_stats=True,
        )
        self.llm_engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.flow.fp16 = fp16
        if self.fp16 is True:
            self.flow.half()
        self.token_hop_len = 2 * self.flow.input_frame_rate
        # here we fix flow encoder/decoder decoding_chunk_size, in the future we will send it as arguments, or use cache
        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = 2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()

        self.mix_ratio = mix_ratio or [5, 15]

        self.lock = asyncio.Lock()  # 改为异步锁

        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

        # 与vllm中的模型保持一致
        self.speech_token_size = 6564
        self.llm_token_size = 151936
        self.sos_eos_token_id = self.speech_token_size + self.llm_token_size + 1
        self.task_token_id = self.sos_eos_token_id + 1
        self.zero_token_id = self.task_token_id + 1

    def load(self, flow_model, hift_model):
        self.flow.load_state_dict(torch.load(flow_model, weights_only=True, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, weights_only=True, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model):
            convert_onnx_to_trt(flow_decoder_estimator_model, flow_decoder_onnx_model, fp16)
        if os.path.getsize(flow_decoder_estimator_model) == 0:
            raise ValueError('{} is empty file, delete it and export again!'.format(flow_decoder_estimator_model))
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        if self.flow.decoder.estimator_engine is None:
            raise ValueError('failed to load trt {}'.format(flow_decoder_estimator_model))
        self.flow.decoder.estimator = self.flow.decoder.estimator_engine.create_execution_context()

    async def llm_inference(self, prompt_token_ids: List[int], request_id: str=None, stop_token_ids=None):
        assert isinstance(prompt_token_ids, list) , "prompt_token_ids should be List[int]"
        invalid = next((i for i, x in enumerate(prompt_token_ids) if not isinstance(x, int)), None)
        assert invalid is None, f"Error in prompt_token_ids, Non-int element at index {invalid}: {prompt_token_ids[invalid]}"
        # print('prompt_token_ids:', prompt_token_ids)
        # TODO: 增加上下文控制，取消请求时
        sampling_params = SamplingParams(
            temperature=0.9,  # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
            top_p=0.95,        # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
            top_k=25,
            # min_tokens=80,       # 不支持设置最小的tokens数量设置，开启后直接崩溃
            # presence_penalty = 1.0,    # 不支持设置
            # frequency_penalty = 0.0,   # 不支持设置
            max_tokens=1024,
            detokenize=False,
            stop_token_ids=stop_token_ids or [6561],
            # stop_token_ids=[6561, 6563],   # for stream
            ignore_eos=False,
            output_kind=RequestOutputKind.DELTA
        )
        async for output in self.llm_engine.generate(
                {
                    "prompt_token_ids": prompt_token_ids,
                },
                sampling_params=sampling_params,
                request_id=request_id or f"{time.time()}",
        ):
            # print(len(output.outputs[0].token_ids), '---', output.outputs[0].token_ids)
            # yield output.outputs[0].token_ids[-1]
            # print(output.outputs[0])
            yield output.outputs[0]


    async def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        prompt_text = tensor_to_list(prompt_text + torch.tensor(6564))
        llm_prompt_speech_token = tensor_to_list(llm_prompt_speech_token)

        if isinstance(text, Generator):
            last_tokens = []
            prompt_token_ids = [self.sos_eos_token_id]
            text_tokens_cache = prompt_text
            for this_text in text:
                this_text = tensor_to_list(this_text + torch.tensor(6564))
                # text need tokens
                assert isinstance(this_text, list), "text need token ids List[int]."
                text_tokens_cache += this_text
                while len(llm_prompt_speech_token) != 0:
                    if len(text_tokens_cache) >= self.mix_ratio[0]:
                        text_input_token = text_tokens_cache[:self.mix_ratio[0]]
                        speech_input_token = llm_prompt_speech_token[:self.mix_ratio[1]]
                        prompt_token_ids += text_input_token + speech_input_token
                        # reset the last cache
                        text_tokens_cache = text_tokens_cache[self.mix_ratio[0]:]
                        llm_prompt_speech_token = llm_prompt_speech_token[self.mix_ratio[1]:]
                    else:
                        logging.info('not enough text token to decode, wait for more')
                        break
                if len(llm_prompt_speech_token) == 0:
                    if (len(last_tokens) > 0 and last_tokens[-1] == 6563) or len(prompt_token_ids) == 1:
                        logging.info('get fill token, need to append more text token')
                        if len(text_tokens_cache) >= self.mix_ratio[0]:
                            text_tokens_temp = text_tokens_cache[:self.mix_ratio[0]]
                            prompt_token_ids += text_tokens_temp
                            logging.info('append {} text token'.format(len(text_tokens_temp)))
                            text_tokens_cache = text_tokens_cache[self.mix_ratio[0]:]
                        else:
                            logging.info('not enough text token to decode, wait for more')
                            continue
                    async for output in self.llm_inference(prompt_token_ids, request_id=uuid, stop_token_ids=[6563]):
                        last_tokens = output.token_ids
                        self.tts_speech_token_dict[uuid].extend(output.token_ids)
                        prompt_token_ids.extend(output.token_ids)
                    # delete the stop token
                    self.tts_speech_token_dict[uuid] = self.tts_speech_token_dict[uuid][:-1]
                    prompt_token_ids = prompt_token_ids[:-1]
            prompt_token_ids += text_tokens_cache + [self.task_token_id]
            logging.info('no more text token, decode until met eos')
            async for output in self.llm_inference(prompt_token_ids, request_id=uuid, stop_token_ids=[6561]):
                self.tts_speech_token_dict[uuid].extend(output.token_ids)
            self.tts_speech_token_dict[uuid] = self.tts_speech_token_dict[uuid][:-1]
        else:
            text = tensor_to_list(text + torch.tensor(6564))
            prompt_token_ids = [self.sos_eos_token_id] + prompt_text + text + \
                               [self.task_token_id] + llm_prompt_speech_token
            async for output in self.llm_inference(prompt_token_ids, request_id=uuid, stop_token_ids=[6561]):
                self.tts_speech_token_dict[uuid].extend(output.token_ids)
            self.tts_speech_token_dict[uuid] = self.tts_speech_token_dict[uuid][:-1]

        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, finalize=False, speed=1.0):
        tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                         token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_token=prompt_token.to(self.device),
                                         prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_feat=prompt_feat.to(self.device),
                                         prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                         embedding=embedding.to(self.device),
                                         finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    async def async_tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        async with self.lock:
            self.tts_speech_token_dict[this_uuid] = []
            self.llm_end_dict[this_uuid] = False
            self.hift_cache_dict[this_uuid] = None
        # queue: asyncio.Queue[int|None] = asyncio.Queue()
        task = asyncio.create_task(self.llm_job(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        if stream is True:
            token_offset = 0
            while True:
                await asyncio.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= self.token_hop_len + self.flow.pre_lookahead_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + self.token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    # this_tts_speech = await asyncio.to_thread(
                    #     self.token2wav,
                    #          token=this_tts_speech_token,
                    #          prompt_token=flow_prompt_speech_token,
                    #          prompt_feat=prompt_speech_feat,
                    #          embedding=flow_embedding,
                    #          uuid=this_uuid,
                    #          token_offset=token_offset,
                    #          finalize=False
                    # )
                    this_tts_speech = self.token2wav(
                             token=this_tts_speech_token,
                             prompt_token=flow_prompt_speech_token,
                             prompt_feat=prompt_speech_feat,
                             embedding=flow_embedding,
                             uuid=this_uuid,
                             token_offset=token_offset,
                             finalize=False
                    )
                    token_offset += self.token_hop_len
                    yield {'tts_speech': this_tts_speech.cpu()}
                if (self.llm_end_dict[this_uuid] is True and
                        len(self.tts_speech_token_dict[this_uuid]) - token_offset < self.token_hop_len + self.flow.pre_lookahead_len):
                    break
            await task
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             token_offset=token_offset,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            await task
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             token_offset=0,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        async with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
