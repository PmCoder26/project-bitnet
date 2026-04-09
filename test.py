from tokenizer.dna_tokenizer import DNATokenizer

tok = DNATokenizer()

dna = "ATGCGTACGTTAG"
print(tok.dna_to_amino_acids(dna))
print(tok.encode(dna))
print(tok.decode(tok.encode(dna)))
