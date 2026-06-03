from model import NGCTransformer
import jax
import jax.numpy as jnp
import numpy as np
from config import Config as config
from data_preprocess.data_loader import DataLoader
from data_preprocess.tokenizer import get_tokenizer, BPETokenizer
from pathlib import Path


# Initialize the model
dkey = jax.random.PRNGKey(0)
model = NGCTransformer(
    dkey, 
    batch_size=config.batch_size, 
    seq_len=config.seq_len, 
    n_embed=config.n_embed, 
    vocab_size=config.vocab_size, 
    n_layers=config.n_layers, 
    n_heads=config.n_heads,
    T=config.n_iter, 
    dt=1., 
    tau_m=config.tau_m, 
    act_fx=config.act_fx, 
    eta=config.eta, 
    dropout_rate=config.dropout_rate, 
    exp_dir="exp",
    loadDir="exp", 
    pos_learnable=config.pos_learnable, 
    optim_type=config.optim_type, 
    wub=config.wub, 
    wlb=config.wlb, 
    model_name="ngc_transformer"
)


tokenizer = get_tokenizer(config)

if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is None:
    vocab_file = getattr(config, "tokenizer_vocab_file", None)
    if vocab_file is None:
        default_path = Path(__file__).parent / "data_preprocess" / "outputs" / "tokenizer" / "bpe_tokenizer.json"
        if default_path.exists():
            vocab_file = str(default_path)
            print(f"Auto-loading BPE tokenizer from default path: {vocab_file}")
    
    # Attempt to load
    if vocab_file and Path(vocab_file).exists():
        tokenizer.load_tokenizer(vocab_file)
        print(f"Loaded BPE tokenizer (vocab size: {tokenizer.get_vocab_size()})")
    else:
        raise RuntimeError(
            "BPE tokenizer not trained or loaded!\n\n"
        )


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    seq_len: int = 8,
    temperature: float = 1.0,
    key=None
):
    """
    Generate text using the model and provided tokenizer.
    Works with both custom BPE and tiktoken backends.
    """
    # Encode prompt - returns jnp.ndarray for both backends
    prompt_ids = tokenizer.encode(prompt)
    
    # Ensure batch dimension: (1, sequence_length)
    if prompt_ids.ndim == 1:
        prompt_tensor = prompt_ids[None, :]
    else:
        prompt_tensor = prompt_ids

    current_tokens = prompt_tensor
    current_key = key

    for _ in range(max_new_tokens):
        # Truncate context to fit model's seq_len
        if current_tokens.shape[1] > seq_len:
            input_seq = current_tokens[:, -seq_len:]
        else:
            input_seq = current_tokens

        # Pad to exactly seq_len if needed (assumes token ID 0 = padding)
        if input_seq.shape[1] < seq_len:
            pad_len = seq_len - input_seq.shape[1]
            input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=0)
        
        # Dummy target for inference (unused when adapt_synapses=False)
        dummy_target = jnp.zeros((config.batch_size * config.seq_len, config.vocab_size))  

        # Forward pass
        y_mu_inf, y_mu, _ = model.process(input_seq, dummy_target, adapt_synapses=False)
        logits = y_mu.reshape(config.batch_size, config.seq_len, config.vocab_size)

        # Get logits for the last *real* token (excluding padding)
        actual_len = min(current_tokens.shape[1], seq_len)
        last_pos = actual_len - 1
        next_logits = logits[0, last_pos, :] / temperature

        # Sample or take argmax
        if current_key is not None:
            probs = jax.nn.softmax(next_logits)
            current_key, subkey = jax.random.split(current_key)
            next_token = jax.random.choice(subkey, a=config.vocab_size, p=probs)
        else:
            next_token = jnp.argmax(next_logits)

        # Append new token
        current_tokens = jnp.concatenate([current_tokens, next_token[None, None]], axis=1)

    # Decode generated IDs back to text
    generated_ids = current_tokens[0].tolist()
    return tokenizer.decode(generated_ids)


# Example usage
if __name__ == "__main__":
    prompt = "The king said: "
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=200,
        seq_len=config.seq_len,        
        temperature=0.8,
        key=jax.random.PRNGKey(42)  
    )
    print(generated)