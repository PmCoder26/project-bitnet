import time
import torch
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import hashlib
import os
import re

from bitnet.model import BitNetDecoder
from data.dataset import UniProtDataset  # not DNADataset

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "./checkpoints/checkpoint_step2500.pth"

# ---------------- FUNCTION NAMES ----------------
FUNCTION_NAMES = [
    "Enzyme",
    "Binding",
    "Transport",
    "Structural",
    "Regulatory",
    "Signal",
    "Unknown",
    "Other"
]

DOMAIN_NAMES = [
    "ATP-binding domain",
    "Kinase domain",
    "Transmembrane region",
    "Zinc finger",
    "SH3 domain",
    "WD repeat"
]

LOCALIZATION_NAMES = [
    "Cytoplasm",
    "Nucleus",
    "Membrane",
    "Mitochondria",
    "Secreted",
    "Extracellular"
]

GO_NAMES = [
    "GO:0004672 (protein kinase activity)",
    "GO:0005524 (ATP binding)",
    "GO:0000166 (nucleotide binding)",
    "GO:0005634 (nucleus)",
    "GO:0005737 (cytoplasm)"
]


# ---------------- LOAD DATASET & MODEL ----------------
dataset = UniProtDataset(tsv_path="data/uniprot_annotations.tsv", max_len=128, max_samples=1000)
tokenizer = dataset.tokenizer


model = BitNetDecoder(
    vocab_size=tokenizer.vocab_size,
    d_model=96,
    n_layers=4,
    n_heads=4,
    max_seq_len=256
)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
# ---------------- EXTRA MODEL STATS ----------------
# Model size in KB
model_size_kb = os.path.getsize(CHECKPOINT_PATH) // 1024

# Extract step from filename (assumes 'checkpoint_stepXXXX.pth')
match = re.search(r'step(\d+)', CHECKPOINT_PATH)
step_number = match.group(1) if match else "—"

# Compute checkpoint hash (SHA1, short 8 chars)
sha1 = hashlib.sha1()
with open(CHECKPOINT_PATH, "rb") as f:
    while True:
        data = f.read(65536)  # read 64kb chunks
        if not data:
            break
        sha1.update(data)
checkpoint_hash = sha1.hexdigest()[:8]

NUM_PARAMS = sum(p.numel() for p in model.parameters())
BITS = 1

# ---------------- FASTAPI ----------------
app = FastAPI(title="BitNet-1 Edge Demo")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------- HTML TEMPLATE ----------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>BitNet-1 | Edge AI Demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

:root {
    --bg:#f8fafc;
    --card:#ffffff;
    --border:#e5e7eb;
    --text:#0f172a;
    --muted:#475569;
    --accent:linear-gradient(135deg,#6366f1,#22c55e);
}

*{box-sizing:border-box;font-family:Inter;}

body{
    margin:0;
    background:linear-gradient(180deg,#f1f5f9,#ffffff);
    display:flex;
    justify-content:center;
    padding:30px;
}

.container{
    width:100%;
    max-width:900px;
    background:var(--card);
    border-radius:24px;
    padding:40px;
    border:1px solid var(--border);
    box-shadow:0 25px 60px rgba(0,0,0,.08);
}

.header{
    display:flex;
    align-items:center;
    gap:14px;
}

.logo{
    width:56px;
    height:56px;
    border-radius:18px;
    background:linear-gradient(135deg,#6366f1,#22c55e);
    display:flex;
    align-items:center;
    justify-content:center;
    color:white;
    font-weight:800;
    letter-spacing:1px;
    box-shadow:0 0 0 rgba(99,102,241,.6);
    animation:pulse 2.5s infinite;
}

.logo span{
    font-size:18px;
}

@keyframes pulse{
    0%{ box-shadow:0 0 0 0 rgba(99,102,241,.6); }
    70%{ box-shadow:0 0 0 18px rgba(99,102,241,0); }
    100%{ box-shadow:0 0 0 0 rgba(99,102,241,0); }
}

.subtitle{
    margin:12px 0 18px;
    color:var(--muted);
}

.badges{
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    margin-bottom:22px;
}

.badge{
    padding:7px 14px;
    border-radius:999px;
    font-size:12px;
    font-weight:600;
    border:1px solid transparent;
    background:#f1f5f9;
    color:#0f172a;
}

.badge.bit{ background:#eef2ff; border-color:#a5b4fc; color:#3730a3; }
.badge.edge{ background:#ecfeff; border-color:#67e8f9; color:#0369a1; }
.badge.power{ background:#f0fdf4; border-color:#86efac; color:#166534; }
.badge.research{ background:#fff7ed; border-color:#fdba74; color:#9a3412; }
.badge.status{ background:#ecfdf5; border-color:#6ee7b7; color:#065f46; }

textarea{
    width:100%;
    min-height:120px;
    padding:16px;
    border-radius:14px;
    border:1px solid var(--border);
    background:#f8fafc;
    font-size:15px;
}

.actions{
    display:flex;
    gap:14px;
    margin-top:18px;
}

button{
    padding:14px;
    border-radius:14px;
    border:none;
    font-weight:600;
    cursor:pointer;
}

.generate{
    background:var(--accent);
    color:white;
    flex:1;
}

.copy{
    background:#f1f5f9;
    border:1px solid var(--border);
}

.output-wrapper {
    margin-top: 15px;
}

.output-container {
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid var(--border);
    padding: 12px 16px 16px 16px;
    font-family: monospace;
    font-size: 14px;
    line-height: 1.3;
    font-weight: 500;
    white-space: pre-wrap;
    min-height: 60px;
    overflow-x: auto;
}

.output-header {
    display: flex;
    justify-content: space-between; /* badge left, copy button right */
    align-items: center;
    margin-bottom: 6px;
}

.output-badge {
    background: linear-gradient(135deg,#6366f1,#22c55e); /* green-blue gradient */
    color: white;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}


.output-text {
    min-height: 60px;
}


/* Compact function bars */
#functionOutput .bar {
    display: grid;
    grid-template-columns: 80px 1fr 35px;
    gap: 2px;
    align-items: center;
    margin-bottom: 2px;   /* less vertical space */
    font-size: 12px;
}
#functionOutput .bar-fill,
#sequenceOutput + .probs .bar-fill {
    height: 4px;
    border-radius: 4px;
}



.probs {
    margin-top: 30px;   /* increased spacing above bars */
}


/* Bars */
.bar{
    display:grid;
    grid-template-columns:60px 1fr 50px;
    gap:8px;
    align-items:center;
    margin-bottom:6px;
}

.bar-fill{
    height:10px;
    border-radius:6px;
    background:linear-gradient(90deg,#6366f1,#22c55e);
}




/* Stats */
.stats{
    margin-top:20px;
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:10px;
    font-size:14px;
    color:var(--muted);
}
.stat-note{
    grid-column:1 / -1;
    font-size:12px;
    color:#64748b;
    margin-top:6px;
}

/* Footer */
.footer{
    margin-top:30px;
    text-align:center;
    font-size:12px;
    color:#64748b;
}
</style>
</head>

<body>
<div class="container">

<div class="header">
<div class="logo"><span>BN</span></div>
<h1>BitNet-1</h1>
</div>

<div class="subtitle">
1-Bit Transformer optimized for ultra-low power Edge Devices
</div>

<div class="badges">
    <span class="badge bit">⚡ 1-Bit Model</span>
    <span class="badge edge">📱 Edge-Optimized</span>
    <span class="badge power">🔋 Ultra-Low Power</span>
    <span class="badge research">🧪 Research Grade</span>
    <span class="badge status">🟢 Live Demo</span>
</div>

<form id="dnaForm">
<textarea name="dna_input" placeholder="Enter DNA / Protein sequence">%%INPUT%%</textarea>
<div class="actions">
<button class="generate">Generate</button>
</div>
</form>

<!-- Outputs -->
<div class="output-wrapper">
  <div class="output-header">
    <span class="output-badge">Sequence Output</span>
    <button class="copy" onclick="copySequence()">Copy</button>
  </div>
  <div id="sequenceOutput" class="output-container">%%OUTPUT%%</div>
</div>

<!-- Function Predictions -->
<div class="output-wrapper">
  <div class="output-header">
    <span class="output-badge">Function Prediction</span>
    <button class="copy" onclick="copyFunctions()">Copy</button>
  </div>
  <div id="functionOutput" class="output-container">%%FUNC_OUTPUT%%</div>
</div>

<!-- Domain Predictions -->
<div class="output-wrapper">
  <div class="output-header">
    <span class="output-badge">Domain Detection</span>
    <button class="copy" onclick="copyDomains()">Copy</button>
  </div>
  <div id="domainOutput" class="output-container">%%DOMAIN_OUTPUT%%</div>
</div>

<!-- GO Term Predictions -->
<div class="output-wrapper">
  <div class="output-header">
    <span class="output-badge">GO Terms</span>
    <button class="copy" onclick="copyGO()">Copy</button>
  </div>
  <div id="goOutput" class="output-container">%%GO_OUTPUT%%</div>
</div>

<!-- Localization Predictions -->
<div class="output-wrapper">
  <div class="output-header">
    <span class="output-badge">Localization</span>
    <button class="copy" onclick="copyLoc()">Copy</button>
  </div>
  <div id="locOutput" class="output-container">%%LOC_OUTPUT%%</div>
</div>



<!-- Top-k probabilities -->
<div class="probs">
<h3>Next-Token Probability</h3>
%%BARS%%
</div>

<!-- Stats -->
<div class="stats">
  <div><b>Parameters:</b> %%PARAMS%%</div>
  <div><b>Precision:</b> %%BITS%%-bit</div>
  <div><b>Device:</b> %%DEVICE%%</div>
  <div><b>Latency:</b> %%LATENCY%% ms</div>
  <div><b>Inference Mode:</b> Greedy</div>
  <div><b>Max Tokens:</b> 50</div>
  <div><b>Input Tokens:</b> %%IN_TOKENS%%</div>
  <div><b>Output Tokens:</b> %%OUT_TOKENS%%</div>

  <div class="stat-note">
    Model Checkpoint: <b>checkpoint_step%%STEP%%.pth</b> 
    (hash: <code>%%HASH%%</code>, size: %%SIZE%% KB)
  </div>
</div>

<div class="footer">
© 2026 • Academic Edge-AI Demonstration
</div>

</div>

<script>
function copySequence(){
    navigator.clipboard.writeText(
        document.getElementById("sequenceOutput").innerText
    );
}

function copyFunctions(){
    navigator.clipboard.writeText(
        document.getElementById("functionOutput").innerText
    );
}
function copyFunctions(){
    navigator.clipboard.writeText(document.getElementById("functionOutput").innerText);
}
function copyDomains(){
    navigator.clipboard.writeText(document.getElementById("domainOutput").innerText);
}
function copyGO(){
    navigator.clipboard.writeText(document.getElementById("goOutput").innerText);
}
function copyLoc(){
    navigator.clipboard.writeText(document.getElementById("locOutput").innerText);
}

const form = document.getElementById("dnaForm");
form.addEventListener("submit", async (e)=>{
    e.preventDefault();

    const fd = new FormData(form);
    const res = await fetch("/", { method:"POST", body:fd });
    const html = await res.text();

    const doc = new DOMParser().parseFromString(html,"text/html");

    // Typewriter effect for sequence
    const newText = doc.getElementById("sequenceOutput").innerText;
    const output = document.getElementById("sequenceOutput");
    output.innerText = "";

    let i = 0;
    function type(){
        if(i < newText.length){
            output.innerText += newText.charAt(i++);
            setTimeout(type, 18);
        }
    }
    type();

    // --- Update outputs safely ---
const seqDiv = doc.getElementById("sequenceOutput");
if(seqDiv){
    const output = document.getElementById("sequenceOutput");
    output.innerText = "";
    let i = 0;
    function type(){
        if(i < seqDiv.innerText.length){
            output.innerText += seqDiv.innerText.charAt(i++);
            setTimeout(type, 18);
        }
    }
    type();
}

const funcDiv = doc.getElementById("functionOutput");
if(funcDiv){
    document.getElementById("functionOutput").innerHTML = funcDiv.innerHTML;
}

const domainDiv = doc.getElementById("domainOutput");
if(domainDiv){
    document.getElementById("domainOutput").innerHTML = domainDiv.innerHTML;
}

const goDiv = doc.getElementById("goOutput");
if(goDiv){
    document.getElementById("goOutput").innerHTML = goDiv.innerHTML;
}

const locDiv = doc.getElementById("locOutput");
if(locDiv){
    document.getElementById("locOutput").innerHTML = locDiv.innerHTML;
}

// Update stats + top-k probabilities
const statsDiv = doc.querySelector(".stats");
if(statsDiv){
    document.querySelector(".stats").innerHTML = statsDiv.innerHTML;
}

const probsDiv = doc.querySelector(".probs");
if(probsDiv){
    document.querySelector(".probs").innerHTML = probsDiv.innerHTML;
}

});
</script>
</body>
</html>
"""


# ---------------- ROUTES ----------------
@app.get("/", response_class=HTMLResponse)
async def home():
    return (HTML_PAGE
        .replace("%%OUTPUT%%","")
        .replace("%%INPUT%%","")
        .replace("%%PARAMS%%",f"{NUM_PARAMS:,}")
        .replace("%%BITS%%",str(BITS))
        .replace("%%DEVICE%%",DEVICE.upper())
        .replace("%%LATENCY%%","—")
        .replace("%%BARS%%","")
        .replace("%%IN_TOKENS%%","—")
        .replace("%%OUT_TOKENS%%","—")
        .replace("%%STEP%%", step_number)
        .replace("%%HASH%%", checkpoint_hash)
        .replace("%%SIZE%%", str(model_size_kb))
        .replace("%%FUNC_OUTPUT%%", "")
        .replace("%%DOMAIN_OUTPUT%%", "")
        .replace("%%GO_OUTPUT%%", "")   
        .replace("%%LOC_OUTPUT%%", "")
    )

@app.post("/", response_class=HTMLResponse)
async def generate(dna_input: str = Form(...)):
    dna_input = dna_input.strip()
    if not dna_input:
        return await home()

    input_ids = torch.tensor(
        [tokenizer.encode(dna_input)], device=DEVICE
    )
    input_token_count = input_ids.shape[1]

    start = time.time()
    with torch.no_grad():
        logits, hidden_states, _ = model(
            input_ids,
            attention_mask=torch.ones_like(input_ids)
        )

        generated = model.generate(
            input_ids, max_new_tokens=50
        )

        func_logits = model.predict_function(
            hidden_states,
            attention_mask=torch.ones_like(input_ids)
        )
        domain_logits = model.predict_domain(
        hidden_states,
        attention_mask=torch.ones_like(input_ids)
        )

        loc_logits = model.predict_localization(
            hidden_states,
            attention_mask=torch.ones_like(input_ids)
        )

        go_logits = model.predict_go(
            hidden_states,
            attention_mask=torch.ones_like(input_ids)
        )

    func_probs = torch.sigmoid(func_logits)[0]
    domain_probs = torch.sigmoid(domain_logits)[0]
    loc_probs = torch.sigmoid(loc_logits)[0]
    go_probs = torch.sigmoid(go_logits)[0]
        # ---------------- DEBUG: Print probabilities ----------------
    print("Function probabilities:", func_probs.tolist())
    print("Domain probabilities:", domain_probs.tolist())
    print("Localization probabilities:", loc_probs.tolist())
    print("GO term probabilities:", go_probs.tolist())

    latency = round((time.time() - start) * 1000, 2)
    output_token_count = generated.shape[1] - input_ids.shape[1]

    decoded = tokenizer.decode(generated[0].tolist())
    output_text = " ".join(decoded) if isinstance(decoded, list) else decoded

    # ---------------- Function predictions ----------------
    report_text = ""

    # ---------------- FUNCTIONS ----------------
    func_report = "=== Predicted Protein Function ===\n"

    for name, prob in zip(FUNCTION_NAMES, func_probs):
        if prob.item() > 0.3:
            func_report += f"• {name:<18} Confidence: {prob.item():.2f}\n"

    report_text += "\n"

    # ---------------- DOMAINS ----------------
    domain_report ="=== Predicted Domains ===\n"

    for name, prob in zip(DOMAIN_NAMES, domain_probs):
        if prob.item() > 0.03:
            domain_report += f"• {name}\n"


    # ---------------- GO TERMS ----------------
    go_report = "=== Predicted GO Terms ===\n"

    for name, prob in zip(GO_NAMES, go_probs):
        if prob.item() > 0.05:
            go_report += f"• {name}\n"


    # ---------------- LOCALIZATION ----------------
    loc_report = "=== Predicted Localization ===\n"

    for name, prob in zip(LOCALIZATION_NAMES, loc_probs):
        if prob.item() > 0.4:
            loc_report += f"• {name}\n"

    if loc_report.strip() == "":
        loc_report = "No confident localization predictions."

    # fallback if no confident function
    if func_report.strip() == "":
        func_report = "• No confident function detected\n"
    domain_report_html = domain_report.replace("\n", "<br>")
    go_report_html = go_report.replace("\n", "<br>")
    loc_report_html = loc_report.replace("\n", "<br>")
    func_report_html = func_report.replace("\n", "<br>")

    # ---------------- Next-token probabilities ----------------

    probs = torch.softmax(logits[:, -1, :], dim=-1)
    topk = torch.topk(probs, 5)

    bars = ""
    for idx, val in zip(topk.indices[0], topk.values[0]):
        tok = tokenizer.decode([idx.item()])
        bars += f"""
        <div class="bar">
            <span>{tok}</span>
            <div class="bar-fill" style="width:{val.item()*100:.1f}%"></div>
            <small>{val.item():.2f}</small>
        </div>
        """

    return (HTML_PAGE
    .replace("%%OUTPUT%%", output_text)
    .replace("%%FUNC_OUTPUT%%", func_report_html)
    .replace("%%DOMAIN_OUTPUT%%", domain_report_html)
    .replace("%%GO_OUTPUT%%", go_report_html)
    .replace("%%LOC_OUTPUT%%", loc_report_html)
    .replace("%%INPUT%%", dna_input)
    .replace("%%PARAMS%%", f"{NUM_PARAMS:,}")
    .replace("%%BITS%%", str(BITS))
    .replace("%%DEVICE%%", DEVICE.upper())
    .replace("%%LATENCY%%", str(latency))
    .replace("%%BARS%%", bars)
    .replace("%%IN_TOKENS%%", str(input_token_count))
    .replace("%%OUT_TOKENS%%", str(output_token_count))
    .replace("%%STEP%%", step_number)
    .replace("%%HASH%%", checkpoint_hash)
    .replace("%%SIZE%%", str(model_size_kb))
)

