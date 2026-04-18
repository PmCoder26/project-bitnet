import time
import json
import torch
import os
import numpy as np
import platform
import subprocess
import re
import psutil

from bitnet.model import BitNetDecoder
from data.dataset import UniProtDataset

# --------------------------------------------------
# PLATFORM DETECTION
# --------------------------------------------------
SYSTEM = platform.system()

# --------------------------------------------------
# FORCE CPU ONLY (FAIR BASELINE)
# --------------------------------------------------
DEVICE = "cpu"
torch.set_num_threads(os.cpu_count())

print(f"Platform: {SYSTEM}")
print(f"Using device: {DEVICE} (FP32 CPU baseline)")


# --------------------------------------------------
# MAC ENERGY MONITOR (REAL POWER)
# --------------------------------------------------
class MacEnergyMonitor:
    def __init__(self):
        self.samples = []

    def begin(self):
        pass

    def end(self):
        try:
            result = subprocess.run(
                [
                    "sudo",
                    "powermetrics",
                    "--samplers",
                    "cpu_power,gpu_power",
                    "-n",
                    "1",
                    "--show-process-energy"
                ],
                capture_output=True,
                text=True,
                timeout=15
            )

            output = result.stdout + "\n" + result.stderr

            cpu_match = re.search(r"CPU\s*[Pp]ower:\s*([0-9.]+)\s*mW", output)
            gpu_match = re.search(r"GPU\s*[Pp]ower:\s*([0-9.]+)\s*mW", output)
            ane_match = re.search(r"ANE\s*[Pp]ower:\s*([0-9.]+)\s*mW", output)

            cpu_power = float(cpu_match.group(1)) if cpu_match else 0.0
            gpu_power = float(gpu_match.group(1)) if gpu_match else 0.0
            ane_power = float(ane_match.group(1)) if ane_match else 0.0

            total_watts = (cpu_power + gpu_power + ane_power) / 1000.0

            if total_watts > 0:
                self.samples.append(total_watts)

        except Exception as e:
            print(f"⚠️ powermetrics error: {e}")

    def average_power(self):
        if not self.samples:
            return None
        return float(sum(self.samples) / len(self.samples))


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
dataset = UniProtDataset(
    "data/uniprot_annotations.tsv",
    max_len=128,
    max_samples=100
)

tokenizer = dataset.tokenizer


# --------------------------------------------------
# LOAD MODEL (FP32 CPU)
# --------------------------------------------------
model = BitNetDecoder(
    vocab_size=tokenizer.vocab_size,
    d_model=96,
    n_layers=4,
    n_heads=4,
    max_seq_len=256
)

checkpoint_path = "checkpoints/checkpoint_step382500.pth"

model.load_state_dict(
    torch.load(checkpoint_path, map_location="cpu")
)

model.to("cpu")
model.eval()

print("✅ FP32 model loaded successfully on CPU")


# --------------------------------------------------
# MEASURE FUNCTION (REAL METRICS)
# --------------------------------------------------
def measure(input_ids):
    mac_monitor = None

    if SYSTEM == "Darwin":
        mac_monitor = MacEnergyMonitor()
        mac_monitor.begin()

    process = psutil.Process(os.getpid())

    start = time.perf_counter()

    with torch.no_grad():
        _ = model(input_ids)

    end = time.perf_counter()

    if mac_monitor:
        mac_monitor.end()

    # -------------------------
    # BASIC METRICS
    # -------------------------
    latency_ms = (end - start) * 1000
    tokens = input_ids.numel()

    memory_mb = process.memory_info().rss / (1024 ** 2)
    throughput = tokens / (latency_ms / 1000)

    # -------------------------
    # POWER + ENERGY (REAL)
    # -------------------------
    power_watts = None
    energy_joules = None
    energy_per_token = None

    if mac_monitor:
        power_watts = mac_monitor.average_power()

        if power_watts is not None:
            energy_joules = power_watts * (latency_ms / 1000)

            if tokens > 0:
                energy_per_token = energy_joules / tokens

    return {
        "latency_ms": latency_ms,
        "memory_mb": memory_mb,
        "throughput_tokens_per_sec": throughput,
        "power_watts": power_watts,
        "energy_joules": energy_joules,
        "energy_per_token": energy_per_token
    }


# --------------------------------------------------
# CLEAN AVERAGE
# --------------------------------------------------
def clean_avg(values):
    values = [v for v in values if v is not None]

    if len(values) == 0:
        return None

    if len(values) < 3:
        return float(sum(values) / len(values))

    threshold = np.percentile(values, 95)
    filtered = [v for v in values if v <= threshold]

    return float(sum(filtered) / len(filtered)) if filtered else None


# --------------------------------------------------
# RUN BENCHMARK
# --------------------------------------------------
def run():
    sample = dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0).to("cpu")

    print("Running warmup...")
    for _ in range(5):
        measure(input_ids)

    print("Starting FP32 benchmark (REAL POWER)...")

    results = []

    for i in range(20):
        result = measure(input_ids)
        results.append(result)

        print(
            f"[FP32] Run {i+1:02d} | "
            f"Latency: {result['latency_ms']:.2f} ms | "
            f"Throughput: {result['throughput_tokens_per_sec']:.2f} tok/s | "
            f"Memory: {result['memory_mb']:.2f} MB | "
            f"Power: {result['power_watts']} W | "
            f"Energy/token: {result['energy_per_token']}"
        )

    # -------------------------
    # FINAL METRICS
    # -------------------------
    output = {
        "model_type": "fp32_cpu",
        "platform": SYSTEM,
        "device": "cpu",

        "avg_latency_ms": clean_avg([r["latency_ms"] for r in results]),
        "avg_memory_mb": clean_avg([r["memory_mb"] for r in results]),
        "avg_throughput_tokens_per_sec": clean_avg([r["throughput_tokens_per_sec"] for r in results]),
        "avg_power_watts": clean_avg([r["power_watts"] for r in results]),
        "avg_energy_joules": clean_avg([r["energy_joules"] for r in results]),
        "avg_energy_per_token": clean_avg([r["energy_per_token"] for r in results]),

        "latency_std_ms": float(np.std([r["latency_ms"] for r in results])),
        "num_runs": len(results),
        "runs": results
    }

    os.makedirs("results", exist_ok=True)

    with open("results/fp32.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\n✅ FP32 benchmark complete (REAL POWER)")
    print("Saved to: results/fp32.json")


if __name__ == "__main__":
    run()