"""
[DESCRIPTION]
Provides a clean, modular interface to interact with locally hosted Ollama language models.

[OUTPUT]
- Console display of both user prompt and model response.
- Saved chat transcript in `chat_logs/` with summarizing and timestamped filename.
- Optionally returns last-layer embeddings or dimensions as NumPy arrays.

Author: Jim Yang
Date: 2025-04-15
"""

import os
import re
import datetime
import numpy as np
from ollama import chat
from llama_cpp import Llama

# ==================== MODEL PATHS ====================
model_paths = {
    "llama2:13b":        "/home/prompt5398/.ollama/models/blobs/sha256-2609048d349e7c70196401be59bea7eb89a968d4642e409b0e798b34403b96c8",  # ~7.3G
    "llama3.2:1b":       "/home/prompt5398/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45",  # ~1.3G
    "gemma2:2b":         "/home/prompt5398/.ollama/models/blobs/sha256-7462734796d67c40ecec2ca98eddf970e171dbb6b370e43fd633ee75b69abe1b",  # ~1.6G
    "phi3":              "/home/prompt5398/.ollama/models/blobs/sha256-633fc5be925f9a484b61d6f9b9a78021eeb462100bd557309f01ba84cac26adf",  # ~2.2G
    "deepseek-r1:1.5b":  "/home/prompt5398/.ollama/models/blobs/sha256-9801e7fce27dbf3d0bfb468b7b21f1d132131a546dfc43e50518631b8b1800a9-partial" # ~1.1G
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_LOG_DIR = os.path.join(SCRIPT_DIR, "chat_logs")
os.makedirs(CHAT_LOG_DIR, exist_ok=True)

# -------------------- TITLE MAKER --------------------
def create_title_from_prompt(user_prompt: str, max_chars: int = 50) -> str:
    """
    Auto-generates a short title from the user's prompt (up to max_chars).
    Replaces non-alphanumeric with underscores. 
    If it ends up empty, defaults to 'NEW_CHAT'.
    """
    summary_prompt = f"""Below is a prompt. 
    Please provide a extremely concise title for this prompt/chat that
    encapsulates this prompt in STRICTLY LESS THAN **5 WORDS** to identify this chat.
    YOU MUST NOT INCLUDE ANYTHING ELSE, no commentary, just summary words.
    ---
    {user_prompt}
    """

    try:
        summary_resp = chat(model=model_name, messages=[{"role": "user", "content": summary_prompt}])
        short_summary = summary_resp["message"]["content"].strip()[:max_chars]
    except Exception as e:
        print(f"[WARNING] Could not fetch short summary: {e}")
        short_summary = "NEW_CHAT"

    short_summary_sanitized = re.sub(r"[^\w]+", "_", short_summary)
    short_summary_sanitized = re.sub(r"^[^A-Za-z0-9]+", "", short_summary_sanitized)

    if not short_summary_sanitized:
        return "NEW_CHAT"

    return short_summary_sanitized

# -------------------- SANITIZE TITLE FUNCTION --------------------
def sanitize_title(title: str) -> str:
    """
    Sanitizes a title string so that it contains only alphanumeric characters and underscores.
    Ensures the first character is a letter or digit. If the result is empty, returns 'NEW_CHAT'.
    """
    sanitized = re.sub(r"[^\w]+", "_", title.strip())
    sanitized = re.sub(r"^[^A-Za-z0-9]+", "", sanitized)
    if not sanitized:
        sanitized = "NEW_CHAT"
    return sanitized

# -------------------- CHAT WITH MODEL (one-shot) --------------------
def chat_with_model(prompt: str, model_name: str, chat_title: str = None) -> str:
    """
    Sends a single prompt to the specified Ollama model (non-stream).
    Prints the entire response and saves it to a .txt file named:
        <chat_title>_<model_name>_<timestamp>.txt
    If chat_title is None, we auto-generate from the user's prompt (up to 50 chars);
    otherwise, the provided chat_title is sanitized.
    """
    if not chat_title:
        chat_title = create_title_from_prompt(prompt)
    else:
        chat_title = sanitize_title(chat_title)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{chat_title}_{model_name}_{timestamp}.txt"
    session_path = os.path.join(CHAT_LOG_DIR, filename)

    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = chat(model=model_name, messages=messages)
        full_response = response["message"]["content"]
    except Exception as e:
        print(f"[ERROR] {e}")
        return ""

    print(f"\n[User]\n{prompt.strip()}\n\n[Model - {model_name}]\n{full_response}")

    with open(session_path, "w") as f:
        f.write(f"[User]\n{prompt.strip()}\n\n[Model ({model_name})]\n{full_response.strip()}\n")

    print(f"\n[SUCCESS] Chat saved to: {session_path}")
    return full_response

# -------------------- EMBEDDINGS FUNCTION (OPTIONAL) --------------------
def get_embeddings(prompt: str, model_path: str) -> np.ndarray:
    """
    Uses llama_cpp to extract embeddings from the last hidden layer.
    Passes verbose=False to suppress extra debug prints from the library.
    """
    llm = Llama(model_path=model_path, embedding=True, verbose=False)
    embedding_list = llm.embed(prompt)
    return np.array(embedding_list)

def model(model_name="llama3.2:1b", prompt="PLEASE ENTER PROMPT"):
    # model_name = "deepseek-r1:1.5b"  # choose from "llama3.2:1b", "llama2:13b", "gemma2:2b", "phi3", "deepseek-r1:1.5b"


    # ==================== MODEL PATHS ====================
    model_paths = {
        "llama2:13b":        "/home/prompt5398/.ollama/models/blobs/sha256-2609048d349e7c70196401be59bea7eb89a968d4642e409b0e798b34403b96c8",  # ~7.3G
        "llama3.2:1b":       "/home/prompt5398/.ollama/models/blobs/sha256-74701a8c35f6c8d9a4b91f3f3497643001d63e0c7a84e085bed452548fa88d45",  # ~1.3G
        "gemma2:2b":         "/home/prompt5398/.ollama/models/blobs/sha256-7462734796d67c40ecec2ca98eddf970e171dbb6b370e43fd633ee75b69abe1b",  # ~1.6G
        "phi3":              "/home/prompt5398/.ollama/models/blobs/sha256-633fc5be925f9a484b61d6f9b9a78021eeb462100bd557309f01ba84cac26adf",  # ~2.2G
        "deepseek-r1:1.5b":  "/home/prompt5398/.ollama/models/blobs/sha256-9801e7fce27dbf3d0bfb468b7b21f1d132131a546dfc43e50518631b8b1800a9-partial" # ~1.1G
    }

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CHAT_LOG_DIR = os.path.join(SCRIPT_DIR, "chat_logs")
    os.makedirs(CHAT_LOG_DIR, exist_ok=True)
    
    model_path = model_paths[model_name]

    ##########################################
    #                                        #
    #               EDIT BELOW               #
    #                                        #
    ##########################################

    custom_title = None # Optionally set a custom chat title.
    final_output = chat_with_model(prompt, model_name, chat_title=custom_title)

    if final_output:
        if False:  # Change to True to print embeddings shape
            embeddings = get_embeddings(prompt, model_path)
            print(f"[INFO] Embeddings shape: {embeddings.shape}")

    return final_output