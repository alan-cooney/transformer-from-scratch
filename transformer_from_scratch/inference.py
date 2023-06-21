"""Model inference"""
import argparse

import torch
from transformers import GPTNeoXTokenizerFast

from transformer_from_scratch.transformer import Transformer

END_OF_TEXT_TOKEN = "<|endoftext|>"


def load_model(checkpoint_path):
    """Load the model."""
    transformer_model = Transformer()
    transformer_model.load_state_dict(torch.load(checkpoint_path))
    transformer_model.eval()
    return transformer_model


def create_tokenizer() -> GPTNeoXTokenizerFast:
    """Create the tokenizer."""
    return GPTNeoXTokenizerFast.from_pretrained("gpt2", pad_token=END_OF_TEXT_TOKEN)


def tokenize_prompt(
    text: str,
) -> str:
    """Tokenize a prompt"""
    tokenizer = create_tokenizer()
    tokenized = tokenizer(
        text,
        padding="max_length",  # Pad to the max length
        truncation=True,  # Truncate to the max length
        max_length=1024,  # 1024 is the default max length for our transformer,
        is_split_into_words=False,
        return_attention_mask=False,
        return_tensors="pt",  # Return a pytorch tensor per prompt
    )
    return tokenized["input_ids"]


def preprocess_input(tokens):
    """Preprocess the input tokens."""
    input_tensor = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    return input_tensor


def postprocess_output(logits):
    """Postprocess the logits."""
    _, batch_predicted_tokens = torch.max(logits, dim=-1)
    batch_predicted_tokens = batch_predicted_tokens.squeeze(
        0
    ).tolist()  # Remove batch dimension
    return batch_predicted_tokens


def inference(transformer_model, prompt_input_tokens):
    """Inference."""
    input_tensor = preprocess_input(prompt_input_tokens)
    with torch.no_grad():
        logits = transformer_model(input_tensor)
    batch_predicted_tokens = postprocess_output(logits)
    return batch_predicted_tokens


if __name__ == "__main__":
    # Load the model checkpoint
    CHECKPOINT_PATH = "./.checkpoints/model_latest.pt"
    model = load_model(CHECKPOINT_PATH)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Transformer Inference")
    parser.add_argument("prompt", type=str, help="Prompt text")
    args = parser.parse_args()
    prompt = args.prompt

    # Tokenize the prompt and generate text
    generated_text = prompt
    input_tokens = tokenize_prompt(prompt)

    gpt_tokenizer = create_tokenizer()

    while len(input_tokens) < 100:
        # Generate the next token
        predicted_indices = inference(model, input_tokens)
        next_token = predicted_indices[-1]

        if next_token == END_OF_TEXT_TOKEN:
            break

        generated_text += gpt_tokenizer.decode([next_token])
        input_tokens.append(next_token)

    print("Generated Text:")
    print(generated_text)
