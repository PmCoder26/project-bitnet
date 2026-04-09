# bitnet/train.py
import math
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast  # or sentencepiece

from .model import BitNetDecoder

def get_optimizer_and_scheduler(model, total_steps, peak_lr=1e-3, warmup_steps=750, weight_decay=0.01):
    optimizer = AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.98), weight_decay=weight_decay)
    # polynomial decay: lr = lr0 * (1 - step/total_steps)^power ; simplest implement with LambdaLR
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1.0, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return (1.0 - progress) ** 1.0  # power = 1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

def train_loop(model, dataloader, device, epochs, total_steps, save_every=1000):
    model.to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model, total_steps)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    step = 0
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits, _ = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                       shift_labels.view(-1), ignore_index=-100)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            step += 1
            if step % save_every == 0:
                torch.save(model.state_dict(), f"checkpoint_step{step}.pth")
            if step >= total_steps:
                return