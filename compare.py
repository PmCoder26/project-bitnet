import json

bitnet = json.load(open("results/bitnet.json"))
fp32 = json.load(open("results/fp32.json"))

def safe_div(a, b):
    if a is None or b in [0, None]:
        return None
    return a / b

comparison = {
    "latency_speedup": safe_div(fp32["avg_latency"], bitnet["avg_latency"]),
    "throughput_gain": safe_div(bitnet["avg_throughput"], fp32["avg_throughput"]),
    "memory_reduction": safe_div(fp32["avg_memory"], bitnet["avg_memory"]),
    "power_efficiency": safe_div(fp32["avg_power_watts"], bitnet["avg_power_watts"]),
    "energy_saving": safe_div(fp32["avg_energy_joules"], bitnet["avg_energy_joules"]),
    "energy_per_token_gain": safe_div(fp32["avg_energy_per_token"], bitnet["avg_energy_per_token"])
}

json.dump(comparison, open("results/comparison.json","w"), indent=4)

print("\n=== FINAL COMPARISON ===")
print(comparison)
