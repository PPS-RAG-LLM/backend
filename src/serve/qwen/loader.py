from transformers import AutoModelForCausalLM, AutoTokenizer

def load_qwen_instruct_7b(model_dir): 
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    return tokenizer, model




