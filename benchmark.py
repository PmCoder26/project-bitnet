import time, json, torch, os, numpy as np

# ---- SAFE pyRAPL SETUP ----
USE_RAPL = True
try:
    import pyRAPL
    pyRAPL.setup()
except Exception as e:
    print("⚠️ pyRAPL not available:", e)
    USE_RAPL = False

from bitnet.model import BitNetDecoder
from data.dataset import UniProtDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = UniProtDataset("data/uniprot_annotations.tsv", max_len=128, max_samples=100)
tokenizer = dataset.tokenizer

model = BitNetDecoder(
    vocab_size=tokenizer.vocab_size,
    d_model=96,
    n_layers=4,
    n_heads=4,
    max_seq_len=256
)

model.load_state_dict(torch.load("./checkpoints/checkpoint_step382500.pth", map_location=DEVICE))
model.to(DEVICE).eval()


def measure(input_ids):
    meter = None

    if USE_RAPL:
        try:
            meter = pyRAPL.Measurement('bitnet')
            meter.begin()
        except:
            meter = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    with torch.no_grad():
        model(input_ids)

    end = time.perf_counter()

    if meter:
        try:
            meter.end()
        except:
            meter = None

    latency = (end - start) * 1000
    tokens = input_ids.numel()

    # ---- SAFE ENERGY ----
    energy = None
    if meter and meter.result and meter.result.pkg:
        energy = meter.result.pkg[0]

    power = energy / (latency / 1000) if energy and latency > 0 else None
    energy_per_token = energy / tokens if energy and tokens > 0 else None

    mem = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
    throughput = tokens / (latency / 1000)

    return latency, mem, throughput, power, energy, energy_per_token


def clean_avg(values):
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return None
    threshold = np.percentile(values, 95)
    filtered = [v for v in values if v <= threshold]
    return sum(filtered) / len(filtered)


def run():
    sample = dataset[0]
    input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)

    results = []

    for _ in range(5):
        measure(input_ids)

    for i in range(20):
        lat, mem, thr, power, energy, ept = measure(input_ids)

        results.append({
            "latency": lat,
            "memory": mem,
            "throughput": thr,
            "power_watts": power,
            "energy_joules": energy,
            "energy_per_token": ept
        })

        print(f"[BitNet] {i}: {lat:.2f} ms | {thr:.2f} tok/s | Power: {power}")

    avg_lat = clean_avg([r["latency"] for r in results])
    avg_mem = clean_avg([r["memory"] for r in results])
    avg_thr = clean_avg([r["throughput"] for r in results])
    avg_power = clean_avg([r["power_watts"] for r in results])
    avg_energy = clean_avg([r["energy_joules"] for r in results])
    avg_ept = clean_avg([r["energy_per_token"] for r in results])

    std_lat = float(np.std([r["latency"] for r in results]))

    out = {
        "type": "bitnet",
        "avg_latency": avg_lat,
        "avg_memory": avg_mem,
        "avg_throughput": avg_thr,
        "avg_power_watts": avg_power,
        "avg_energy_joules": avg_energy,
        "avg_energy_per_token": avg_ept,
        "latency_std": std_lat,
        "runs": results
    }

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/bitnet.json", "w"), indent=4)


if __name__ == "__main__":
    run()
