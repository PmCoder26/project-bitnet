import time
import json
import torch
import os
import numpy as np
import platform
import subprocess
import re
import threading
import psutil

from bitnet.model import BitNetDecoder
from data.dataset import UniProtDataset

# --------------------------------------------------
# PLATFORM DETECTION
# --------------------------------------------------
SYSTEM = platform.system()

# --------------------------------------------------
# DEVICE SETUP
# --------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Platform: {SYSTEM}")
print(f"Using device: {DEVICE}")

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

            cpu_match = re.search(
                r"CPU\s*[Pp]ower:\s*([0-9.]+)\s*mW",
                output
            )

            gpu_match = re.search(
                r"GPU\s*[Pp]ower:\s*([0-9.]+)\s*mW",
                output
            )

            package_match = re.search(
                r"Package\s*[Pp]ower:\s*([0-9.]+)\s*mW",
                output
            )

            ane_match = re.search(
                r"ANE\s*[Pp]ower:\s*([0-9.]+)\s*mW",
                output
            )

            combined_match = re.search(
                r"Combined\s*[Pp]ower.*?:\s*([0-9.]+)\s*mW",
                output
            )

            cpu_power = float(cpu_match.group(1)) if cpu_match else 0.0
            gpu_power = float(gpu_match.group(1)) if gpu_match else 0.0
            package_power = float(package_match.group(1)) if package_match else 0.0
            ane_power = float(ane_match.group(1)) if ane_match else 0.0

            total_mw = cpu_power + gpu_power + ane_power

            if total_mw > 0:
                self.samples.append(total_mw / 1000.0)
            elif package_power > 0:
                self.samples.append(package_power / 1000.0)
            elif combined_match:
                self.samples.append(float(combined_match.group(1)) / 1000.0)

        except Exception as e:
            print(f"⚠️ powermetrics parse error: {e}")

    def average_power(self):
        if len(self.samples) == 0:
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
# LOAD MODEL
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
    torch.load(checkpoint_path, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

print("✅ Model loaded successfully")


# --------------------------------------------------
# SINGLE BENCHMARK MEASUREMENT
# --------------------------------------------------
def measure(input_ids):
    meter = None
    mac_monitor = None

    if USE_RAPL and pyRAPL is not None:
        try:
            meter = pyRAPL.Measurement("bitnet")
            meter.begin()
        except Exception:
            meter = None

    elif USE_MAC_ENERGY:
        mac_monitor = MacEnergyMonitor()
        mac_monitor.begin()

    # Device synchronization
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    elif DEVICE == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        _ = model(input_ids)

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    elif DEVICE == "mps":
        torch.mps.synchronize()

    end = time.perf_counter()

    if meter is not None:
        try:
            meter.end()
        except Exception:
            meter = None

    if mac_monitor is not None:
        mac_monitor.end()

    latency_ms = (end - start) * 1000
    tokens = input_ids.numel()

    energy_joules = None
    power_watts = None
    energy_per_token = None

    # Linux energy calculation
    if meter is not None:
        try:
            if meter.result and meter.result.pkg:
                energy_microjoules = meter.result.pkg[0]
                energy_joules = energy_microjoules / 1_000_000

                if latency_ms > 0:
                    power_watts = energy_joules / (latency_ms / 1000)

                if tokens > 0:
                    energy_per_token = energy_joules / tokens
        except Exception:
            pass

    # macOS energy calculation
    elif mac_monitor is not None:
        power_watts = mac_monitor.average_power()

        if power_watts is not None:
            energy_joules = power_watts * (latency_ms / 1000)

            if tokens > 0:
                energy_per_token = energy_joules / tokens

    # Memory measurement
    if DEVICE == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    elif DEVICE == "mps":
        try:
            memory_mb = torch.mps.current_allocated_memory() / (1024 ** 2)
        except Exception:
            memory_mb = None

    else:
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

    if len(filtered) == 0:
        filtered = values

    return float(sum(filtered) / len(filtered))


# --------------------------------------------------
# MAIN BENCHMARK
# --------------------------------------------------
def run():
    sample = dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)

    print(f"Input shape: {tuple(input_ids.shape)}")
    print("Running warmup...")

    for _ in range(5):
        measure(input_ids)

    print("Starting benchmark...")

    results = []

    for i in range(20):
        result = measure(input_ids)
        results.append(result)

        line = (
            f"[BitNet] Run {i + 1:02d} | "
            f"Latency: {result['latency_ms']:.2f} ms | "
            f"Throughput: {result['throughput_tokens_per_sec']:.2f} tok/s"
        )

        if result["memory_mb"] is not None:
            line += f" | Memory: {result['memory_mb']:.2f} MB"

        if result["power_watts"] is not None:
            line += f" | Power: {result['power_watts']:.2f} W"

        if result["energy_per_token"] is not None:
            line += f" | Energy/token: {result['energy_per_token']:.8f} J"

        print(line)

    avg_latency = clean_avg([r["latency_ms"] for r in results])
    avg_memory = clean_avg([r["memory_mb"] for r in results])
    avg_throughput = clean_avg([r["throughput_tokens_per_sec"] for r in results])
    avg_power = clean_avg([r["power_watts"] for r in results])
    avg_energy = clean_avg([r["energy_joules"] for r in results])
    avg_energy_per_token = clean_avg([r["energy_per_token"] for r in results])

    latency_std = float(np.std([r["latency_ms"] for r in results]))

    output = {
        "model_type": "bitnet",
        "platform": SYSTEM,
        "device": DEVICE,
        "rapl_enabled": USE_RAPL,
        "mac_energy_enabled": USE_MAC_ENERGY,
        "input_shape": list(input_ids.shape),
        "avg_latency_ms": avg_latency,
        "avg_memory_mb": avg_memory,
        "avg_throughput_tokens_per_sec": avg_throughput,
        "avg_power_watts": avg_power,
        "avg_energy_joules": avg_energy,
        "avg_energy_per_token": avg_energy_per_token,
        "latency_std_ms": latency_std,
        "num_runs": len(results),
        "runs": results
    }

    os.makedirs("results", exist_ok=True)

    with open("results/bitnet.json", "w") as f:
        json.dump(output, f, indent=4)

    print("\n✅ Benchmark complete")
    print("Results saved to: results/bitnet.json")

    print("\n===== SUMMARY =====")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Average Throughput: {avg_throughput:.2f} tok/s")

    if avg_memory is not None:
        print(f"Average Memory: {avg_memory:.2f} MB")

    if avg_power is not None:
        print(f"Average Power: {avg_power:.2f} W")

    if avg_energy is not None:
        print(f"Average Energy: {avg_energy:.6f} J")

    if avg_energy_per_token is not None:
        print(f"Average Energy per Token: {avg_energy_per_token:.8f} J")

    print(f"Latency Std Dev: {latency_std:.2f} ms")


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    run()