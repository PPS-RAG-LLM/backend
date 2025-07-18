from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.generation.configuration_utils import GenerationConfig

def load_qwen_instruct_7b(model_dir): 
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    # gen_config = GenerationConfig.from_pretrained(model_dir)
    return model, tokenizer




