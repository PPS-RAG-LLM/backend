from .logger import logger
from .time import  now_kst_string, now_kst
from .slug import generate_unique_slug, generate_thread_slug, make_safe_filename
from .middleware import ProcessTimeMiddleware
from .database import get_db, init_db
from .vaildator import validate_category_subcategory, validate_category
from .model_load import get_or_load_embedder_async

def free_torch_memory(sync: bool = True) -> None:
	"""지연 로딩: torch 의존성이 필요한 시점에만 임포트"""
	from .memory import free_torch_memory as _free
	return _free(sync=sync)


def load_embedding_model():
	"""지연 로딩: transformers/torch 의존성이 필요한 시점에만 임포트"""
	from .model_load import load_embedding_model as _load
	return _load()

__all__ = [
	"logger", 
	"now_kst",
	"now_kst_string",
	"generate_unique_slug", 
	"generate_thread_slug",
	"ProcessTimeMiddleware",
	"get_db", 
	"init_db",
	"free_torch_memory",
	"load_embedding_model",	
	"validate_category_subcategory",
	"validate_category",
	"make_safe_filename",
	"get_or_load_embedder_async",
	]       