import ollama

# --- ENVIRONMENT CONFIGURATION ---
sl_segment_input = "Hello, how are you doing today? I hope you are having a great time."

# --- OLLAMA SERVER & MODEL SETTINGS ---
MODEL = "gemma4:e4b"
HOST = "http://localhost:11434"
TIMEOUT = 5  # Note: Connection timeout is usually managed at the client level

# --- LLM GENERATION PARAMETERS (Ollama Options) ---
generation_params = {
    "temperature": 0.0,
    "num_predict": 128,
    "num_ctx": 2048,
    "repeat_penalty": 1.2,
    "top_k": 40,
    "top_p": 0.9,
    "seed": 42,
    "stop": ["\n", "###"]
}

# --- LLM PROMPT AND RESPONSE PARSING ---
# We prepare the prompt by replacing the {SLsegment} variable
prompt_template = f"Translate the following text from English into Spanish. Provide only the translation.\nEnglish: {sl_segment_input}"

def executePrompt(prompt):
    try:
        # We initialize the client with the specified host
        client = ollama.Client(host=HOST)
        
        # We make the call to the generation API (generate)
        # You could also use client.chat if it were a message structure
        response = client.generate(
            model=MODEL,
            prompt=prompt,
            options=generation_params
        )
        
        # We extract the text from the response
        output_text = response['response']
        
        # --- POST-PROCESSING (json_key & regex_pattern) ---
        # In this case, since json_key is "None" and regex_pattern is None,
        # we directly print the raw result of the response.
        print("--- Gemma 4 answer ---")
        print(output_text)
        
    except Exception as e:
        print(f"Error in connection or generation: {e}")

if __name__ == "__main__":
    prompt="Translate the following text from English into Spanish. Provide only the translation.\nEnglish: This an example of translation using a large language model."
    executePrompt(prompt)
