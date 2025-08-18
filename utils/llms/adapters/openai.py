# utils/llms/adapters/openai.py
from utils.llms.registry import register, Streamer
from utils.llms.openai.streamer import stream_chat as openai_stream

class OpenAIStreamer:
	def __init__(self, model: str):
		self.model = model
	def stream(self, messages, **kw):
		return openai_stream(messages, model=self.model, **kw)

@register("openai")
def openai_factory(model_key: str) -> Streamer:
	return OpenAIStreamer(model_key)