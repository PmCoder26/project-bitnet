import torch
from bitnet.model import BitNetDecoder
from data.dataset import UniProtDataset

# -------------------------
# Settings (must match training)
# -------------------------
D_MODEL = 96
N_LAYERS = 4
N_HEADS = 4
MAX_SEQ_LEN = 256
CHECKPOINT_PATH = "./checkpoints/checkpoint_step150.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Dataset (for tokenizer only)
# -------------------------
dataset = UniProtDataset(
    fasta_path="data/uniprot_sprot.fasta",
    max_len=128,
    max_samples=10
)

tokenizer = dataset.tokenizer

# -------------------------
# Load Model
# -------------------------
model = BitNetDecoder(
    vocab_size=tokenizer.vocab_size,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    max_seq_len=MAX_SEQ_LEN
).to(device)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# -------------------------
# Prediction Keywords
# -------------------------
FUNCTION_KEYWORDS = ["Kinase", "Transporter", "Enzyme"]
DOMAIN_KEYWORDS = ["Zinc Finger", "Helicase"]
LOCALIZATION_KEYWORDS = ["Nucleus", "Cytoplasm"]
GO_KEYWORDS = ["ATP binding", "DNA binding"]

# -------------------------
# Inference Function
# -------------------------
def pretty_print_predictions(model, tokenizer, sequence):
    with torch.no_grad():
        ids = torch.tensor([tokenizer.encode(sequence)], device=device)
        logits, hidden, _ = model(ids)

        func = torch.sigmoid(model.predict_function(hidden))
        dom = torch.sigmoid(model.predict_domain(hidden))
        loc = torch.sigmoid(model.predict_localization(hidden))
        go = torch.sigmoid(model.predict_go(hidden))

        print("\n=== Predicted Protein Function ===")
        for i, score in enumerate(func[0]):
            if score > 0.5:
                print(f"{FUNCTION_KEYWORDS[i]} | Confidence: {score:.2f}")

        print("\n=== Predicted Domains ===")
        for i, score in enumerate(dom[0]):
            if score > 0.5:
                print(f"{DOMAIN_KEYWORDS[i]} | Confidence: {score:.2f}")

        print("\n=== Predicted Localization ===")
        for i, score in enumerate(loc[0]):
            if score > 0.5:
                print(f"{LOCALIZATION_KEYWORDS[i]} | Confidence: {score:.2f}")

        print("\n=== Predicted GO Terms ===")
        for i, score in enumerate(go[0]):
            if score > 0.5:
                print(f"{GO_KEYWORDS[i]} | Confidence: {score:.2f}")

# -------------------------
# Test Sequence
# -------------------------
test_sequence = "MKTAYIAKQRQISFVKSHFSRQDILD"

pretty_print_predictions(model, tokenizer, test_sequence)
