from torch.utils.data import DataLoader
from bitnet.model import BitNetDecoder
from data.dataset import DNADataset
import torch

PAD_TOKEN_ID = 0

def collate_fn(batch, pad_token_id=PAD_TOKEN_ID):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    lengths = [len(x) for x in input_ids]
    max_len = max(lengths)
    batch_size = len(batch)

    padded_inputs = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=torch.long
    )

    padded_labels = torch.full(
        (batch_size, max_len),
        -100,  # ignore index
        dtype=torch.long
    )

    attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long
    )

    for i, (inp, lab) in enumerate(zip(input_ids, labels)):
        seq_len = len(inp)
        padded_inputs[i, :seq_len] = inp
        padded_labels[i, :seq_len] = lab
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": attention_mask
    }


# ---- DATA ----
dataset = DNADataset(num_samples=2000)
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

# ---- MODEL ----
model = BitNetDecoder(
    vocab_size=dataset.tokenizer.vocab_size,
    d_model=96,
    n_layers=4,
    n_heads=4,
    max_seq_len=128
)

# ---- TEST FORWARD + LOSS ----
for batch in loader:
    logits, _ = model(
        batch["input_ids"],
        attention_mask=batch["attention_mask"]
    )

    # logits: [B, T, V]
    # labels: [B, T]

    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch["labels"].view(-1),
        ignore_index=-100
    )

    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
    break
