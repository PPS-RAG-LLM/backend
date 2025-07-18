import os
import re
import yaml
from dotenv import load_dotenv
load_dotenv()

def env_var_constructor(loader, node):
    value = loader.construct_scalar(node)
    if value.startswith("${") and value.endswith("}"):
        env_var = value[2:-1]
        return os.getenv(env_var, "")
    return value

yaml.add_implicit_resolver('!ENV', re.compile(r'^\$\{[^}]+\}$'), Loader=yaml.SafeLoader)
yaml.add_constructor('!ENV', env_var_constructor, Loader=yaml.SafeLoader)

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# 사용 예시
# print(config["qwen"]["model_path"])
# print(config["openai"]["api_key"])