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
# FORCE CPU ONLY
# --------------------------------------------------
DEVICE = "cpu"
torch.set_num_threads(os.cpu_count())  # use all CPU cores

print(f"Platform: {SYSTEM}")
print(f"Using device: {DEVICE} (CPU ONLY)")

# --------------------------------------------------
# ENERGY MEASUREMENT FLAGS
# --------------------------------------------------
USE_RAPL = False
USE_MAC_ENERGY = False
pyRAPL = None

if SYSTEM == "Linux":
    try:
        import pyRAPL
        pyRAPL.setup()
        USE_RAPL = True
        print("✅ Linux RAPL energy measurement enabled")
    except Exception as e:
        print(f"⚠️ Linux RAPL unavailable: {e}")

elif SYSTEM == "Darwin":
    USE_MAC_ENERGY = True
    print("✅ macOS powermetrics energy measurement enabled")

else:
    print("⚠️ Energy measurement not supported on this OS")


# --------------------------------------------------
# MAC ENERGY MONITOR
# --------------------------------------------------
class MacEnergyMonitor:
    def __init__(self):
        self.samples = []

    def begin(self):
        self.start_time = time.perf_counter()

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
            package_match = re.search(r"Package\s*[Pp]ower:\s*([0-9.]+)\s*mW", output)
            ane_match = re.search(r"ANE\s*[Pp]ower:\s*([0-9.]+)\s*mW", output)

            cpu_power = float(cpu_match.group(1)) if cpu_match else 0.0
            gpu_power = float(gpu_match.group(1)) if gpu_match else 0.0
            ane_power = float(ane_match.group(1)) if ane_match else 0.0
            package_power = float(package_match.group(1)) if package_match else 0.0

            total_mw = cpu_power + gpu_power + ane_power

            if total_mw > 0:
                self.samples.append(total_mw / 1000.0)
            elif package_power > 0:
                self.samples.append(package_power / 1000.0)

        except Exception as e:
            print(f"⚠️ powermetrics parse error: {e}")

    def average_power(self):
        return None if len(self.samples) == 0 else float(sum(self.samples) / len(self.samples))


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
# LOAD MODEL (CPU ONLY)
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

print("✅ Model loaded on CPU")


# --------------------------------------------------
# MEASURE FUNCTION (CPU ONLY)
# --------------------------------------------------
def measure(input_ids):
    meter = None
    mac_monitor = None

    if USE_RAPL and pyRAPL is not None:
        try:
            meter = pyRAPL.Measurement("bitnet")
            meter.begin()
        except:
            meter = None

    elif USE_MAC_ENERGY:
        mac_monitor = MacEnergyMonitor()
        mac_monitor.begin()

    start = time.perf_counter()

    with torch.no_grad():
        _ = model(input_ids)

    end = time.perf_counter()

    if meter is not None:
        try:
            meter.end()
        except:
            meter = None

    if mac_monitor is not None:
        mac_monitor.end()

    latency_ms = (end - start) * 1000
    tokens = input_ids.numel()

    power_watts = None
    energy_joules = None
    energy_per_token = None

    if meter is not None:
        try:
            if meter.result and meter.result.pkg:
                energy_joules = meter.result.pkg[0] / 1_000_000
                power_watts = energy_joules / (latency_ms / 1000)
                energy_per_token = energy_joules / tokens if tokens else None
        except:
            pass

    elif mac_monitor is not None:
        power_watts = mac_monitor.average_power()
        if power_watts is not None:
            energy_joules = power_watts * (latency_ms / 1000)
            energy_per_token = energy_joules / tokens if tokens else None

    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 ** 2)

    throughput = tokens / (latency_ms / 1000)

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

    print("Starting CPU benchmark...")

    results = []

    for i in range(20):
        result = measure(input_ids)
        results.append(result)

        print(
            f"[CPU] Run {i+1:02d} | "
            f"Latency: {result['latency_ms']:.2f} ms | "
            f"Throughput: {result['throughput_tokens_per_sec']:.2f} tok/s"
        )

    avg_latency = clean_avg([r["latency_ms"] for r in results])
    avg_memory = clean_avg([r["memory_mb"] for r in results])
    avg_throughput = clean_avg([r["throughput_tokens_per_sec"] for r in results])

    output = {
        "model_type": "bitnet_cpu",
        "platform": SYSTEM,
        "device": "cpu",
        "avg_latency_ms": avg_latency,
        "avg_memory_mb": avg_memory,
        "avg_throughput_tokens_per_sec": avg_throughput,
        "runs": results
    }

    os.makedirs("results", exist_ok=True)

    with open("results/bitnet_cpu.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\n✅ CPU benchmark complete")


if __name__ == "__main__":
    run()