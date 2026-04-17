import json
import os

# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------
bitnet = json.load(open("../results/bitnet.json"))
fp32 = json.load(open("../results/fp32.json"))

os.makedirs("../results", exist_ok=True)

# --------------------------------------------------
# SAFE DIVISION
# --------------------------------------------------
def safe_div(a, b):
    if a is None or b in [0, None]:
        return None
    return a / b


# --------------------------------------------------
# COMPARISON METRICS
# --------------------------------------------------
comparison = {
    # Performance
    "latency_speedup": safe_div(
        fp32.get("avg_latency_ms"),
        bitnet.get("avg_latency_ms")
    ),

    "throughput_gain": safe_div(
        bitnet.get("avg_throughput_tokens_per_sec"),
        fp32.get("avg_throughput_tokens_per_sec")
    ),

    # Memory efficiency
    "memory_reduction": safe_div(
        fp32.get("avg_memory_mb"),
        bitnet.get("avg_memory_mb")
    ),

    # Energy efficiency (may be None on macOS if power missing)
    "power_efficiency": safe_div(
        fp32.get("avg_power_watts"),
        bitnet.get("avg_power_watts")
    ),

    "energy_saving": safe_div(
        fp32.get("avg_energy_joules"),
        bitnet.get("avg_energy_joules")
    ),

    "energy_per_token_gain": safe_div(
        fp32.get("avg_energy_per_token"),
        bitnet.get("avg_energy_per_token")
    ),

    # Stability metric (extra useful)
    "latency_stability_gain": safe_div(
        fp32.get("latency_std_ms"),
        bitnet.get("latency_std_ms")
    )
}

# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------
with open("../results/comparison.json", "w") as f:
    json.dump(comparison, f, indent=4)

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------
print("\n=== FINAL COMPARISON ===")
for k, v in comparison.items():
    print(f"{k}: {v}")