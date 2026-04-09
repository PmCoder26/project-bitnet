import torch
from torch.utils.data import Dataset
from tokenizer.dna_tokenizer import DNATokenizer
import csv

FUNCTION_KEYWORDS = [
    "kinase",
    "transferase",
    "hydrolase",
    "transcription",
    "receptor",
    "binding",
    "membrane",
    "transport"
]

DOMAIN_KEYWORDS = [
    "atp-binding",
    "kinase domain",
    "transmembrane",
    "zinc finger",
    "sh3",
    "wd repeat"
]

LOCALIZATION_KEYWORDS = [
    "cytoplasm",
    "nucleus",
    "membrane",
    "mitochondr",
    "secreted",
    "extracellular"
]

GO_KEYWORDS = [
    "go:0004672",
    "go:0005524",
    "go:0000166",
    "go:0005634",
    "go:0005737",
]


class UniProtDataset(Dataset):
    # ✅ CHANGE 1: Default max_samples set to None
    def __init__(self, tsv_path, max_len=256, max_samples=None):

        self.tokenizer = DNATokenizer()
        self.samples = []
        self.max_len = max_len

        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            # print(reader.fieldnames)  # Optional: uncomment if you need to debug headers

            count = 0
            for row in reader:

                # ✅ CHANGE 2: Only break if max_samples is NOT None and limit is reached
                if max_samples is not None and count >= max_samples:
                    break

                sequence = row.get("Sequence", "")
                if not sequence:
                    continue

                metadata = self._combine_metadata(row)

                self._add_sample(sequence, metadata)
                count += 1

        print(f"Loaded {len(self.samples)} protein samples from TSV.")

    def _combine_metadata(self, row):
        text_fields = [
            row.get("Protein names", ""),
            row.get("Function [CC]", ""),
            row.get("Binding site", ""),
            row.get("Active site", ""),
            row.get("Domain [CC]", ""),
            row.get("Subcellular location [CC]", ""),
            row.get("Gene Ontology (biological process)", ""),
            row.get("Gene Ontology (cellular component)", ""),
            row.get("Gene Ontology (molecular function)", ""),
            row.get("Gene Ontology (GO)", ""),
            row.get("Gene Ontology IDs", "")
        ]

        combined = " ".join(text_fields).lower()
        return combined

    def _extract_multi_labels(self, text, keywords):
        labels = torch.zeros(len(keywords))
        for i, keyword in enumerate(keywords):
            if keyword in text:
                labels[i] = 1.0
        return labels

    def _add_sample(self, sequence, metadata):

        sequence = sequence[:self.max_len]
        token_ids = self.tokenizer.encode(sequence)

        if len(token_ids) < 3:
            return

        input_ids = token_ids[:-1]
        labels = token_ids[1:]

        function_labels = self._extract_multi_labels(metadata, FUNCTION_KEYWORDS)
        domain_labels = self._extract_multi_labels(metadata, DOMAIN_KEYWORDS)
        loc_labels = self._extract_multi_labels(metadata, LOCALIZATION_KEYWORDS)
        go_labels = self._extract_multi_labels(metadata, GO_KEYWORDS)

        self.samples.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "function_labels": function_labels,
            "domain_labels": domain_labels,
            "loc_labels": loc_labels,
            "go_labels": go_labels
        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]