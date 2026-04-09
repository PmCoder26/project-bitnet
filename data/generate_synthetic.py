# data/generate_synthetic.py
import random

BASES = ["A", "T", "G", "C"]

def random_dna(length_codons=30):
    dna = "".join(random.choice(BASES) for _ in range(length_codons * 3))
    return dna
