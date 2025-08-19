# utils/llms/adapters/qwen.py
from config import config
from utils.llms.registry import register, Streamer
from utils.llms.huggingface.qwen import qwen_7b, qwen_vl_7b
from utils.llms.huggingface.openai import gpt_oss_20b
from utils import logger

logger = logger(__name__)

def _resolve_hf_key(model_key: str):
	# 별칭 → 실제 키 매핑
	reg = (config.get("registry")).get("huggingface")
	logger.info(f"reg: {reg}")
	info = reg.get(model_key)
	logger.info(f"info: {info}")
	return model_key, info

class _Wrap:
	def __init__(self, fn): self.fn = fn
	def stream(self, messages, **kw): return self.fn(messages, **kw)

@register("huggingface")
def hf_factory(model_key: str) -> Streamer:
	key, info = _resolve_hf_key(model_key)
	family = info.get("family")
	local_path = info.get("local_path")
	logger.info(f"hf_factory: {model_key}, {family}, {local_path}")

	if not family:
		# 별칭/레지스트리 모두에서 못 찾은 케이스
		raise ValueError(f"지원하지 않는 HF 모델: {model_key}")

	if family == "qwen":
		return _Wrap(lambda messages, **kw: qwen_7b.stream_chat(messages, model_path=local_path, **kw))
	if family == "qwen-vl":
		return _Wrap(lambda messages, **kw: qwen_vl_7b.stream_chat(messages, model_path=local_path, **kw))
	if family == "openai":
		return _Wrap(lambda messages, **kw: gpt_oss_20b.stream_chat(messages, model_path=local_path, **kw))
	raise ValueError(f"지원하지 않는 HF 모델 패밀리: {family}")