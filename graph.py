import json
import matplotlib.pyplot as plt

bitnet = json.load(open("results/bitnet.json"))
fp32 = json.load(open("results/fp32.json"))

# ---- SAFE HELPER ----
def safe(v):
    return 0 if v is None else v

# ---- Extract data ----
bit_lat = [r["latency"] for r in bitnet["runs"] if r["latency"] is not None]
fp_lat = [r["latency"] for r in fp32["runs"] if r["latency"] is not None]

bit_thr = [r["throughput"] for r in bitnet["runs"] if r["throughput"] is not None]
fp_thr = [r["throughput"] for r in fp32["runs"] if r["throughput"] is not None]

# ---- 1. LATENCY BAR ----
plt.figure()
plt.bar(
    ["BitNet", "FP32"],
    [safe(bitnet.get("avg_latency")), safe(fp32.get("avg_latency"))]
)
plt.title("Average Latency (ms)")
plt.savefig("results/latency_bar.png")
plt.close()

# ---- 2. BOX PLOT ----
plt.figure()
plt.boxplot([bit_lat, fp_lat], labels=["BitNet", "FP32"])
plt.title("Latency Distribution")
plt.savefig("results/latency_box.png")
plt.close()

# ---- 3. LINE PLOT ----
plt.figure()
plt.plot(bit_lat, label="BitNet")
plt.plot(fp_lat, label="FP32")
plt.legend()
plt.title("Latency Over Runs")
plt.savefig("results/latency_line.png")
plt.close()

# ---- 4. THROUGHPUT ----
plt.figure()
plt.bar(
    ["BitNet", "FP32"],
    [safe(bitnet.get("avg_throughput")), safe(fp32.get("avg_throughput"))]
)
plt.title("Throughput (tokens/sec)")
plt.savefig("results/throughput.png")
plt.close()

# ---- 5. POWER ----
plt.figure()
plt.bar(
    ["BitNet", "FP32"],
    [safe(bitnet.get("avg_power_watts")), safe(fp32.get("avg_power_watts"))]
)
plt.title("Power Consumption (Watts)")
plt.savefig("results/power.png")
plt.close()

# ---- 6. ENERGY ----
plt.figure()
plt.bar(
    ["BitNet", "FP32"],
    [safe(bitnet.get("avg_energy_joules")), safe(fp32.get("avg_energy_joules"))]
)
plt.title("Total Energy (Joules)")
plt.savefig("results/energy.png")
plt.close()

# ---- 7. ENERGY PER TOKEN ----
plt.figure()
plt.bar(
    ["BitNet", "FP32"],
    [safe(bitnet.get("avg_energy_per_token")), safe(fp32.get("avg_energy_per_token"))]
)
plt.title("Energy per Token")
plt.savefig("results/energy_per_token.png")
plt.close()

print("✅ All graphs generated safely!")
