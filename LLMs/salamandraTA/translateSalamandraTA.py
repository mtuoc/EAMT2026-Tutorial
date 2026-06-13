import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def translate_salamandra_manual(text, model_id, source_lang, target_lang):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16
    )

    # 1. Manual construction of the prompt following the model card format
    # We define a System Prompt to provide context to the model
    system_prompt = "You are a professional translator."
    user_prompt = f"Translate the following text from {source_lang} into {target_lang}.\n{source_lang}: {text} \n{target_lang}:"

    # We assemble the parts using the <|im_start|> and <|im_end|> delimiters
    full_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # 2. Tokenization
    inputs = tokenizer(full_prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_length = inputs.input_ids.shape[1]

    # 3. Generation
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=400,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>") # We force it to know when to stop
    )

    # 4. Decoding
    result = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # We clean up any possible remnants of the final delimiter if present
    return result.replace("<|im_end|>", "").strip()

if __name__ == "__main__":
    #model_id = "BSC-LT/salamandraTA-2b-instruct"
    model_id = "../../models/salamandraTA-2b-instruct"
    # Usage example
    src = "English"
    tgt = "Spanish"
    frase = "Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making."

    traduccio = translate_salamandra_manual(frase, model_id, src, tgt)
    print(f"\nTranslation: {traduccio}")
