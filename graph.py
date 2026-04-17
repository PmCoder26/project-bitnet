import json
import os
import matplotlib.pyplot as plt

# --------------------------------------------------
# LOAD RESULTS
# --------------------------------------------------
with open("results/bitnet.json", "r") as f:
    bitnet = json.load(f)

with open("results/fp32.json", "r") as f:
    fp32 = json.load(f)

os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# SAFE HELPER
# --------------------------------------------------
def safe(v):
    return 0 if v is None else v


# --------------------------------------------------
# EXTRACT RUN DATA
# --------------------------------------------------
bit_lat = [
    r["latency_ms"]
    for r in bitnet["runs"]
    if r["latency_ms"] is not None
]

fp_lat = [
    r["latency_ms"]
    for r in fp32["runs"]
    if r["latency_ms"] is not None
]

bit_thr = [
    r["throughput_tokens_per_sec"]
    for r in bitnet["runs"]
    if r["throughput_tokens_per_sec"] is not None
]

fp_thr = [
    r["throughput_tokens_per_sec"]
    for r in fp32["runs"]
    if r["throughput_tokens_per_sec"] is not None
]

bit_mem = [
    r["memory_mb"]
    for r in bitnet["runs"]
    if r["memory_mb"] is not None
]

fp_mem = [
    r["memory_mb"]
    for r in fp32["runs"]
    if r["memory_mb"] is not None
]

# --------------------------------------------------
# 1. LATENCY BAR
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.bar(
    ["BitNet", "FP32"],
    [
        safe(bitnet.get("avg_latency_ms")),
        safe(fp32.get("avg_latency_ms"))
    ]
)
plt.ylabel("Milliseconds")
plt.title("Average Latency")
plt.savefig("results/latency_bar.png")
plt.close()

# --------------------------------------------------
# 2. LATENCY DISTRIBUTION BOX PLOT
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.boxplot([bit_lat, fp_lat], labels=["BitNet", "FP32"])
plt.ylabel("Milliseconds")
plt.title("Latency Distribution")
plt.savefig("results/latency_box.png")
plt.close()

# --------------------------------------------------
# 3. LATENCY OVER RUNS
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(bit_lat, label="BitNet")
plt.plot(fp_lat, label="FP32")
plt.xlabel("Run")
plt.ylabel("Latency (ms)")
plt.title("Latency Over Runs")
plt.legend()
plt.savefig("results/latency_line.png")
plt.close()

# --------------------------------------------------
# 4. THROUGHPUT BAR
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.bar(
    ["BitNet", "FP32"],
    [
        safe(bitnet.get("avg_throughput_tokens_per_sec")),
        safe(fp32.get("avg_throughput_tokens_per_sec"))
    ]
)
plt.ylabel("Tokens / sec")
plt.title("Average Throughput")
plt.savefig("results/throughput_bar.png")
plt.close()

# --------------------------------------------------
# 5. THROUGHPUT OVER RUNS
# --------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(bit_thr, label="BitNet")
plt.plot(fp_thr, label="FP32")
plt.xlabel("Run")
plt.ylabel("Tokens / sec")
plt.title("Throughput Over Runs")
plt.legend()
plt.savefig("results/throughput_line.png")
plt.close()

# --------------------------------------------------
# 6. MEMORY USAGE
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.bar(
    ["BitNet", "FP32"],
    [
        safe(bitnet.get("avg_memory_mb")),
        safe(fp32.get("avg_memory_mb"))
    ]
)
plt.ylabel("MB")
plt.title("Average Memory Usage")
plt.savefig("results/memory_bar.png")
plt.close()

# --------------------------------------------------
# 7. POWER CONSUMPTION
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.bar(
    ["BitNet", "FP32"],
    [
        safe(bitnet.get("avg_power_watts")),
        safe(fp32.get("avg_power_watts"))
    ]
)
plt.ylabel("Watts")
plt.title("Average Power Consumption")
plt.savefig("results/power_bar.png")
plt.close()

# --------------------------------------------------
# 8. TOTAL ENERGY
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.bar(
    ["BitNet", "FP32"],
    [
        safe(bitnet.get("avg_energy_joules")),
        safe(fp32.get("avg_energy_joules"))
    ]
)
plt.ylabel("Joules")
plt.title("Average Energy Consumption")
plt.savefig("results/energy_bar.png")
plt.close()

# --------------------------------------------------
# 9. ENERGY PER TOKEN
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.bar(
    ["BitNet", "FP32"],
    [
        safe(bitnet.get("avg_energy_per_token")),
        safe(fp32.get("avg_energy_per_token"))
    ]
)
plt.ylabel("Joules / Token")
plt.title("Energy Per Token")
plt.savefig("results/energy_per_token_bar.png")
plt.close()

# --------------------------------------------------
# 10. LATENCY STD DEV
# --------------------------------------------------
plt.figure(figsize=(6, 5))
plt.bar(
    ["BitNet", "FP32"],
    [
        safe(bitnet.get("latency_std_ms")),
        safe(fp32.get("latency_std_ms"))
    ]
)
plt.ylabel("Milliseconds")
plt.title("Latency Standard Deviation")
plt.savefig("results/latency_std_bar.png")
plt.close()

print("✅ All graphs generated successfully!")
print("Saved inside results/")