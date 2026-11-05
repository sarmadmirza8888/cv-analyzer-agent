"""Local Mistral model: discovery, loading, raw generation, and the agent-facing LLM wrapper."""
import glob
import os

import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from cv_agent.config import KAGGLE_INPUT_DIR, MAX_NEW_TOKENS, MODEL_PATH_ENV, STOP_STRINGS


def find_model_path() -> str:
    """CV_AGENT_MODEL_PATH if set, otherwise search /kaggle/input for a Mistral folder."""
    override = os.environ.get(MODEL_PATH_ENV)
    if override:
        return override

    candidates = glob.glob(f"{KAGGLE_INPUT_DIR}/**/config.json", recursive=True)

    if not candidates:
        raise FileNotFoundError(
            f"No model folder found under {KAGGLE_INPUT_DIR}. "
            f"Attach the Mistral model dataset, or set {MODEL_PATH_ENV}."
        )

    mistral_candidates = [c for c in candidates if "mistral" in c.lower()]
    chosen = mistral_candidates[0] if mistral_candidates else candidates[0]

    return os.path.dirname(chosen)


def load_model(model_path: str):
    """Load tokenizer and model. Returns (tokenizer, model)."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading Mistral model (this can take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
    )

    print("Mistral model loaded successfully!")
    return tokenizer, model


def chat_generate(tokenizer, model, messages, max_new_tokens: int, **generate_kwargs) -> str:
    """Apply the chat template, generate, and decode only the new tokens."""
    # return_dict=True gives back a proper BatchEncoding with input_ids +
    # attention_mask, which we unpack explicitly with **. Relying on
    # return_tensors="pt" alone is ambiguous across transformers versions
    # (it can hand back a raw tensor OR a dict-like object depending on the
    # tokenizer's chat template), which is what caused the "shape" KeyError.
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

    input_length = encoded["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def ask_mistral(tokenizer, model, prompt: str, max_new_tokens: int = 200) -> str:
    """Quick sanity-check helper: raw prompt in, raw text out."""
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": prompt},
    ]
    return chat_generate(
        tokenizer, model, messages, max_new_tokens,
        temperature=0.7, top_p=0.9, do_sample=True,
    )


def build_llm(model, tokenizer):
    """Wrap the model for LangChain agents. Returns (pipeline, llm)."""
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
        stop_strings=STOP_STRINGS,
    )
    text_generation_pipeline._forward_params["tokenizer"] = tokenizer

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return text_generation_pipeline, llm
