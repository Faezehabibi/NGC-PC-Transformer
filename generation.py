from model import NGCTransformer
import jax
import jax.numpy as jnp
import numpy as np
from config import Config as config
from data_preprocess.data_loader import DataLoader
from data_preprocess.tokenizer import get_tokenizer, BPETokenizer
from pathlib import Path
import re
import textwrap





def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    seq_len: int = config.seq_len,
    temperature: float = 1.0,
    top_k: int = 0,
    key=None,
    pad_token_id: int = None
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

    if pad_token_id is None:
        if isinstance(tokenizer, BPETokenizer) and tokenizer.tokenizer is not None:
            pad_token_id = tokenizer.tokenizer.token_to_id("<pad>")
        else:
            pad_token_id = 0

    for _ in range(max_new_tokens):
        # Truncate context to fit model's seq_len
        if current_tokens.shape[1] > config.seq_len:
            input_seq = current_tokens[:, -config.seq_len:]
        else:
            input_seq = current_tokens

        # Pad to exactly seq_len if needed
        if input_seq.shape[1] < config.seq_len:
            pad_len = config.seq_len - input_seq.shape[1]
            input_seq = jnp.pad(input_seq, ((0, 0), (0, pad_len)), constant_values=pad_token_id)
        
        # Forward pass (no target clamping during inference)
        dummy_target = jnp.zeros((config.batch_size * config.seq_len, config.vocab_size))

        # Forward pass

        y_mu_inf, y_mu, _ = model.process(input_seq, dummy_target, adapt_synapses=False)
        logits = y_mu_inf.reshape(config.batch_size, config.seq_len, config.vocab_size)

        # Get logits for the last *real* token (excluding padding)
        if current_tokens.shape[1] > config.seq_len:
            last_pos = config.seq_len - 1
        else:
            last_pos = current_tokens.shape[1] - 1
        next_logits = logits[0, last_pos, :] / temperature

        # Sample or take argmax
        if current_key is not None:
            if top_k is not None and top_k > 0:
                top_k = min(top_k, config.vocab_size)
                top_vals, top_idx = jax.lax.top_k(next_logits, k=top_k)
                probs = jax.nn.softmax(top_vals)
                current_key, subkey = jax.random.split(current_key)
                choice = jax.random.choice(subkey, a=top_k, p=probs)
                next_token = top_idx[choice]
            else:
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


# Initialize the model and tokenizer only when run as a script
if __name__ == "__main__":
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
        loadDir="exp", # Ensure model is loaded from trained exp/ directory
        pos_learnable=config.pos_learnable, 
        optim_type=config.optim_type, 
        wub=config.wub, 
        wlb=config.wlb, 
        model_name="ngc_transformer"
    )

    # Optional: add custom weight stats here if needed

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

    prompt_1 = (
        "First Citizen:\n"
        "Before we proceed any further, hear me speak.\n\n"
        "All:\n"
        "Speak, speak.\n\n"
        "First Citizen:\n"
        "You are all resolved rather to die than to famish?\n\n"
        "All:\n"
        "Resolved. resolved.\n\n"
        "First Citizen:\n"
        "First, you know Caius Marcius is chief enemy to the people.\n\n"
        "All:\n"
        "We know't, we know't."
    )

    prompt_2 = (
        "First Soldier:\n"
        "Fool-hardiness; not I.\n\n"
        "Second Soldier:\n"
        "Nor I.\n\n"
        "First Soldier:\n"
        "See, they have shut him in.\n\n"
        "All:\n"
        "To the pot, I warrant him.\n\n"
        "LARTIUS:\n"
        "What is become of Marcius?\n\n"
        "All:\n"
        "Slain, sir, doubtless.\n\n"
        "First Soldier:\n"
        "Following the fliers at the very heels,\n"
        "With them he enters; who, upon the sudden,\n"
        "Clapp'd to their gates: he is himself alone,\n"
        "To answer all the city.\n\n"
        "LARTIUS:\n"
        "O noble fellow!\n"
        "Who sensibly outdares his senseless sword,\n"
        "And, when it bows, stands up. Thou art left, Marcius:\n"
        "A carbuncle entire, as big as thou art,\n"
        "Were not so rich a jewel. Thou wast a soldier\n"
        "Even to Cato's wish, not fierce and terrible\n"
        "Only in strokes; but, with thy grim looks and\n"
        "The thunder-like percussion of thy sounds,\n"
        "Thou madst thine enemies shake, as if the world\n"
        "Were feverous and did tremble.\n\n"
        "First Soldier:\n"
        "Look, sir.\n\n"
        "LARTIUS:\n"
        "O,'tis Marcius!\n"
        "Let's fetch him off, or make remain alike.\n\n"
        "First Roman:\n"
        "This will I carry to Rome.\n\n"
        "Second Roman:\n"
        "And I this.\n\n"
        "Third Roman:\n"
        "A murrain on't! I took this for silver."
    )

    rng = jax.random.PRNGKey(0)
    rng, key_1 = jax.random.split(rng)
    rng, key_2 = jax.random.split(rng)

    print("\n**************** PROMPT 1 ****************")
    print(prompt_1)
    print("\nFINAL GENERATED 1:\n")
    generated_1 = generate_text(
        model,
        tokenizer,
        prompt_1,
        max_new_tokens=100,
        temperature=0.9,
        top_k=50,
        key=key_1,
    )
    print(generated_1)

    print("\n**************** PROMPT 2 ****************")
    print(prompt_2)
    print("\nFINAL GENERATED 2:\n")
    generated_2 = generate_text(
        model,
        tokenizer,
        prompt_2,
        max_new_tokens=100,
        temperature=0.9,
        top_k=50,
        key=key_2,
    )
    print(generated_2)