# train_full.py
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
from bitnet.model import BitNetDecoder
from data.dataset import UniProtDataset
import os

# -------------------------
# Settings
# -------------------------
BATCH_SIZE = 8
NUM_SAMPLES = 2000  # for testing, increase for real training
D_MODEL = 96
N_LAYERS = 4
N_HEADS = 4
MAX_SEQ_LEN = 256
EPOCHS = 10
LEARNING_RATE = 1e-3
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
CLIP_GRAD = 1.0
SAVE_EVERY = 500  # steps
PAD_TOKEN_ID = 0
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------
# Collate Function
# -------------------------
def collate_fn(batch, pad_token_id=PAD_TOKEN_ID):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    function_labels = [item["function_labels"] for item in batch]

    # ✅ STEP 2 — NEW LABELS ADDED
    domain_labels = [item["domain_labels"] for item in batch]
    loc_labels = [item["loc_labels"] for item in batch]
    go_labels = [item["go_labels"] for item in batch]

    max_len = max(len(x) for x in input_ids)
    batch_size = len(batch)

    padded_inputs = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (inp, lab) in enumerate(zip(input_ids, labels)):
        seq_len = len(inp)
        padded_inputs[i, :seq_len] = inp
        padded_labels[i, :seq_len] = lab
        attention_mask[i, :seq_len] = 1

    # Stack all multi-label tensors
    function_labels = torch.stack(function_labels)
    domain_labels = torch.stack(domain_labels)
    loc_labels = torch.stack(loc_labels)
    go_labels = torch.stack(go_labels)

    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": attention_mask,
        "function_labels": function_labels,
        "domain_labels": domain_labels,
        "loc_labels": loc_labels,
        "go_labels": go_labels
    }


# -------------------------
# Dataset & DataLoader
# -------------------------
dataset = UniProtDataset(
    tsv_path="data/uniprot_annotations.tsv",   # <-- your TSV file
    max_len=128,
    max_samples=10000
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)
print(dataset.tokenizer.vocab_size)


# -------------------------
# Model
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BitNetDecoder(
    vocab_size=dataset.tokenizer.vocab_size,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    max_seq_len=MAX_SEQ_LEN
).to(device)

# -------------------------
# Optimizer & Scheduler
# -------------------------
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), weight_decay=WEIGHT_DECAY)

def lr_lambda(step):
    if step < WARMUP_STEPS:
        return float(step) / float(max(1, WARMUP_STEPS))
    return max(0.0, 1.0 - float(step - WARMUP_STEPS) / float(max(1, EPOCHS * len(loader) - WARMUP_STEPS)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -------------------------
# Mixed precision
# -------------------------
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

# -------------------------
# Training Loop
# -------------------------
step = 0
model.train()
for epoch in range(EPOCHS):
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits, hidden_states, _ = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            func_logits = model.predict_function(
                hidden_states,
                batch["attention_mask"]
            )
            domain_logits = model.predict_domain(hidden_states, batch["attention_mask"])
            loc_logits = model.predict_localization(hidden_states, batch["attention_mask"])
            go_logits = model.predict_go(hidden_states, batch["attention_mask"])

            shift_logits = logits[:, :-1, :]
            shift_labels = batch["labels"][:, 1:]
            token_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100
            )

            func_loss = F.binary_cross_entropy_with_logits(
                func_logits,
                batch["function_labels"]
            )
            domain_loss = F.binary_cross_entropy_with_logits(
                domain_logits,
                batch["domain_labels"]
            )

            loc_loss = F.binary_cross_entropy_with_logits(
                loc_logits,
                batch["loc_labels"]
            )

            go_loss = F.binary_cross_entropy_with_logits(
                go_logits,
                batch["go_labels"]
            )

            loss = (
                token_loss
                + 0.3 * func_loss
                + 0.2 * domain_loss
                + 0.2 * loc_loss
                + 0.2 * go_loss
            )


        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
            optimizer.step()

        scheduler.step()
        step += 1

        if step % 50 == 0:
            print(
                f"Epoch {epoch+1} | Step {step} | "
                f"Total: {loss.item():.4f} | "
                f"Token: {token_loss.item():.4f} | "
                f"Func: {func_loss.item():.4f}"
            )


        if step % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_step{step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

print("Training finished!")
