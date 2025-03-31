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
import re
import json
import inflect
from functools import partial
from collections import OrderedDict
from typing import Generator, Optional, AsyncGenerator, Union, Callable

import librosa
import onnxruntime
import torch
import numpy as np
import whisper
import torchaudio
import torchaudio.compliance.kaldi as kaldi

from pydantic import BaseModel, ConfigDict

from async_cosyvoice.config import OVERWRITE_NORMALIZER_CACHE

try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use WeTextProcessing instead")
    from tn.chinese.normalizer import Normalizer as ZhNormalizer
    from tn.english.normalizer import Normalizer as EnNormalizer
    use_ttsfrd = False
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph, is_only_punctuation


class AsyncTextGeneratorWrpper:
    def __init__(self, obj):
        self.obj = obj
        self.finish = False
        self.history_str = ""  # 这部分记录所有迭代器已经计算了的
        self.history_str_len = 0
        self.cache_str = ""  # 这部分记录的是还没有计算的
        self.cache_str_len = 0
        self.this_history = ""  # 这部分记录的是当前迭代器已经计算了的
        self.this_history_len = 0
        self.is_async_generator = isinstance(obj, AsyncGenerator)

    async def __aiter__(self):
        while True:
            try:
                if self.cache_str_len >= 512:
                    this_str = ""
                    this_str_len = 0
                else:
                    if self.is_async_generator:
                        this_str = await self.obj.__anext__()
                    else:
                        this_str = self.obj.__next__()
                    this_str_len = len(this_str)

                self.cache_str += this_str
                self.cache_str_len += this_str_len

                """
                需要考虑对中文、英文、日文字符串进行流式的句子切分，需要确保当前返回的句子加上 self.this_history 的长度不能超过 512，并且当长度大于 128 时，就需要尽可能的根据标点符号进行切割。
                """

                if self.this_history_len + self.cache_str_len > 128:  # 当长度大于 128 时，尽可能的选择在标点符号的位置进行切割。
                    max_split_len = 512 - self.this_history_len
                    max_pos = min(self.cache_str_len, max_split_len)
                    split_pos = -1
                    split_chars = {'。', '！', '？', '.', '!', '?', '，', '、', '；', ';', ','}

                    # 从后往前找标点符号
                    for i in range(max_pos, 0, -1):
                        if self.cache_str[i - 1] in split_chars:
                            split_pos = i
                            break

                    if split_pos > 0:
                        yield_str = self.cache_str[:split_pos]

                        # 更新变量
                        self.history_str += yield_str
                        self.history_str_len += split_pos

                        logging.debug(f"this history sentence: {self.this_history}")

                        self.this_history = ""
                        self.this_history_len = 0

                        self.cache_str = self.cache_str[split_pos:]
                        self.cache_str_len = len(self.cache_str)

                        yield yield_str

                        return

                if self.this_history_len + self.cache_str_len > 512:  # 这里改为分句逻辑，流式的根据 this_history 进行句子切分，当长度大于 512 时，强制截断到 512 长度

                    need_str_len = 512 - self.this_history_len
                    yield self.cache_str[:need_str_len]
                    self.history_str += self.cache_str[:need_str_len]
                    self.history_str_len += need_str_len

                    self.cache_str = self.cache_str[need_str_len:]
                    self.cache_str_len = len(self.cache_str)

                    logging.debug(f"this history sentence: {self.this_history}")
                    self.this_history_len = 0
                    self.this_history = ""
                    return

                yield self.cache_str
                self.history_str += self.cache_str
                self.history_str_len += self.cache_str_len
                self.this_history += self.cache_str
                self.this_history_len += self.cache_str_len
                self.cache_str = ""
                self.cache_str_len = 0

            except StopIteration:
                self.finish = True
                return
            except StopAsyncIteration:
                self.finish = True
                return



class SpeakerInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str] = None
    spk_id: str
    prompt_text: str
    prompt_text_token: torch.Tensor
    speech_feat: torch.Tensor
    speech_token: torch.Tensor
    embedding: torch.Tensor


class LRUCache(OrderedDict):
    """LRU缓存容器，继承自OrderedDict"""

    def __init__(self, max_size=100_000):
        super().__init__()
        self.max_size = max_size

    def __getitem__(self, key):
        # 访问时移动到末尾（表示最新）
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        # 插入时检查容量，超限则移除最旧项
        if key in self:
            self.move_to_end(key)
        else:
            if len(self) >= self.max_size:
                self.popitem(last=False)
        super().__setitem__(key, value)


class CosyVoiceFrontEnd:

    def __init__(self,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 campplus_model: str,
                 speech_tokenizer_model: str,
                 spk2info: str = '',
                 allowed_special: str = 'all'):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        self.campplus_session = onnxruntime.InferenceSession(campplus_model, sess_options=option, providers=["CPUExecutionProvider"])
        self.speech_tokenizer_session = onnxruntime.InferenceSession(speech_tokenizer_model, sess_options=option,
                                                                     providers=["CUDAExecutionProvider" if torch.cuda.is_available() else
                                                                                "CPUExecutionProvider"])

        self.spk2info = LRUCache(max_size=10000)
        if os.path.exists(spk2info):
            spk_infos = torch.load(spk2info, map_location=self.device, weights_only=False)
            for spk_id, info in spk_infos.items():
                self.spk2info[spk_id] = info
        self.spk2info_path = os.path.join(os.path.dirname(os.path.abspath(spk2info)), 'spk2info')
        os.makedirs(self.spk2info_path, exist_ok=True)
        self.allowed_special = allowed_special
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            assert self.frd.initialize('{}/../pretrained_models/CosyVoice-ttsfrd/resource'.format(ROOT_DIR)) is True, \
                'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyinvg')
        else:
            # self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=True)
            self.zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=OVERWRITE_NORMALIZER_CACHE)
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()

    def _extract_text_token(self, text):
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will return _extract_text_token_generator!')
            # NOTE add a dummy text_token_len for compatibility
            return self._extract_text_token_generator(text), torch.tensor([0], dtype=torch.int32)
        elif isinstance(text, Union[AsyncGenerator, AsyncTextGeneratorWrpper]):
            logging.info('get tts_text async generator, will return _async_extract_text_token_generator!')
            # NOTE add a dummy text_token_len for compatibility
            return self._async_extract_text_token_generator(text), torch.tensor([0], dtype=torch.int32)
        else:
            text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
            text_token = torch.tensor([text_token], dtype=torch.int32)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32)
            return text_token, text_token_len

    async def _async_extract_text_token_generator(self, text_generator):
        async for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i: i + 1]

    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i: i + 1]

    def _extract_speech_token(self, speech):
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(None,
                                                         {self.speech_tokenizer_session.get_inputs()[0].name:
                                                          feat.detach().cpu().numpy(),
                                                          self.speech_tokenizer_session.get_inputs()[1].name:
                                                          np.array([feat.shape[2]], dtype=np.int32)})[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(None,
                                              {self.campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, Union[Generator, AsyncGenerator]):
            logging.info('get tts_text generator, will skip text_normalize!')
            def _text_generator():
                text_generator = AsyncTextGeneratorWrpper(text)
                while True:
                    if text_generator.finish:
                        return
                    yield text_generator
            return _text_generator()

        if text_frontend is False:
            return [text] if split is True else text

        text = text.strip()
        if self.use_ttsfrd:
            texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
            text = ''.join(texts)
        else:
            if contains_chinese(text):
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                if not split:
                    return text
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                if not split:
                    return text
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                             token_min_n=60, merge_len=20, comma_split=False))
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text

    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        self.load_spk_info(spk_id)
        embedding = self.spk2info[spk_id]['embedding']
        assert embedding is not None
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len, 'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, resample_rate):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        if resample_rate == 24000:
            # cosyvoice2, force speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {'text': tts_text_token, 'text_len': tts_text_token_len,
                       'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                       'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                       'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                       'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                       'llm_embedding': embedding, 'flow_embedding': embedding}
        return model_input

    def frontend_cross_lingual(self, tts_text, prompt_speech_16k, resample_rate):
        model_input = self.frontend_zero_shot(tts_text, '', prompt_speech_16k, resample_rate)
        # in cross lingual mode, we remove prompt in llm
        del model_input['prompt_text']
        del model_input['prompt_text_len']
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        # in instruct mode, we remove spk_embedding in llm due to information leakage
        del model_input['llm_embedding']
        instruct_text_token, instruct_text_token_len = self._extract_text_token(instruct_text + '<endofprompt>')
        model_input['prompt_text'] = instruct_text_token
        model_input['prompt_text_len'] = instruct_text_token_len
        return model_input

    def frontend_instruct2(self, tts_text, instruct_text, prompt_speech_16k, resample_rate):
        model_input = self.frontend_zero_shot(tts_text, instruct_text + '<|endofprompt|>', prompt_speech_16k, resample_rate)
        del model_input['llm_prompt_speech_token']
        del model_input['llm_prompt_speech_token_len']
        return model_input

    def frontend_vc(self, source_speech_16k, prompt_speech_16k, resample_rate):
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(prompt_speech_16k)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        source_speech_token, source_speech_token_len = self._extract_speech_token(source_speech_16k)
        model_input = {'source_speech_token': source_speech_token, 'source_speech_token_len': source_speech_token_len,
                       'flow_prompt_speech_token': prompt_speech_token, 'flow_prompt_speech_token_len': prompt_speech_token_len,
                       'prompt_speech_feat': prompt_speech_feat, 'prompt_speech_feat_len': prompt_speech_feat_len,
                       'flow_embedding': embedding}
        return model_input

    def generate_spk_info(self, spk_id: str, prompt_text: str, prompt_speech_16k: torch.Tensor, resample_rate:int=24000, name: str=None):
        assert isinstance(spk_id, str)
        prompt_text_token, _ = self._extract_text_token(prompt_text)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        speech_feat, _ = self._extract_speech_feat(prompt_speech_resample)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        if resample_rate == 24000:
            # cosyvoice2, force speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat = speech_feat[:, :2 * token_len]
            speech_token = speech_token[:, :token_len]
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        spk_info = SpeakerInfo(
            name=name,
            spk_id=spk_id,
            prompt_text=prompt_text,
            prompt_text_token=prompt_text_token,
            speech_feat=speech_feat,
            speech_token=speech_token,
            embedding=embedding,
        )
        self.add_spk_info(spk_id, spk_info)

    def add_spk_info(self, spk_id: str, spk_info: Union[dict, SpeakerInfo]):
        if isinstance(spk_info, BaseModel):
            spk_info = spk_info.model_dump()
        self.spk2info[spk_id] = spk_info
        torch.save(self.spk2info[spk_id], os.path.join(self.spk2info_path, spk_id + '.pt'))

    def load_spk_info(self, spk_id: str):
        if spk_id not in self.spk2info:
            spk_info_path = os.path.join(self.spk2info_path, spk_id + '.pt')
            if not os.path.exists(spk_info_path):
                raise ValueError(f'not found spk2info: {spk_id}')
            spk_info = torch.load(spk_info_path, map_location=self.device, weights_only=False)
            self.spk2info[spk_id] = spk_info

    def delete_spk_info(self, spk_id: str):
        if spk_id in self.spk2info:
            del self.spk2info[spk_id]
        if os.path.exists(os.path.join(self.spk2info_path, spk_id + '.pt')):
            os.remove(os.path.join(self.spk2info_path, spk_id + '.pt'))

    def frontend_instruct2_by_spk_id(self, tts_text, instruct_text, spk_id):
        self.load_spk_info(spk_id)
        tts_text_token, _ = self._extract_text_token(tts_text)
        prompt_text_token, _ = self._extract_text_token(instruct_text + '<|endofprompt|>')
        model_input = {'text': tts_text_token,
                       'prompt_text': prompt_text_token,
                       'flow_prompt_speech_token': self.spk2info[spk_id]['speech_token'],
                       'prompt_speech_feat': self.spk2info[spk_id]['speech_feat'],
                       'llm_embedding': self.spk2info[spk_id]['embedding'],
                       'flow_embedding': self.spk2info[spk_id]['embedding'],
        }
        return model_input

    def frontend_zero_shot_by_spk_id(self, tts_text, spk_id):
        self.load_spk_info(spk_id)
        tts_text_token, _ = self._extract_text_token(tts_text)
        model_input = {'text': tts_text_token,
                       'prompt_text': self.spk2info[spk_id]['prompt_text_token'],
                       'llm_prompt_speech_token': self.spk2info[spk_id]['speech_token'],
                       'flow_prompt_speech_token': self.spk2info[spk_id]['speech_token'],
                       'prompt_speech_feat': self.spk2info[spk_id]['speech_feat'],
                       'llm_embedding': self.spk2info[spk_id]['embedding'],
                       'flow_embedding': self.spk2info[spk_id]['embedding']
        }
        return model_input