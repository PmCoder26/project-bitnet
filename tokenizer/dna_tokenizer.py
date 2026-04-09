# tokenizer/dna_tokenizer.py
from typing import List

class DNATokenizer:
    """
    Now works as PROTEIN tokenizer.
    Keeps same class name so other code doesn't break.
    """

    def __init__(self):
        self.PAD = "<PAD>"
        self.BOS = "<BOS>"
        self.EOS = "<EOS>"
        self.UNK = "<UNK>"

        # 20 standard amino acids
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

        self.tokens = [
            self.PAD,
            self.BOS,
            self.EOS,
            self.UNK,
            *amino_acids
        ]

        self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @property
    def vocab_size(self):
        return len(self.tokens)

    def encode(self, protein_seq: str) -> List[int]:
        protein_seq = protein_seq.strip().upper()

        ids = [self.token_to_id[self.BOS]]

        for aa in protein_seq:
            ids.append(self.token_to_id.get(aa, self.token_to_id[self.UNK]))

        ids.append(self.token_to_id[self.EOS])
        return ids

    def decode(self, token_ids: List[int]) -> List[str]:
        tokens = []
        for i in token_ids:
            t = self.id_to_token.get(i, self.UNK)
            if t in {self.BOS, self.EOS, self.PAD}:
                continue
            tokens.append(t)
        return tokens
