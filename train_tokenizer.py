from pathlib import Path

from tokenizers import ByteLevelBPETokenizer, BertWordPieceTokenizer

#paths = [str(x) for x in Path(".").glob("**/*.txt")]

paths = ["./data/train.txt", "./data/eval.txt"]
#paths = ["/home/rohola/codes/persian_language_models/oscar.eo.txt"]

# Initialize a tokenizer
#tokenizer = ByteLevelBPETokenizer()
tokenizer = BertWordPieceTokenizer()


# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2)

tokenizer.save("FaBerto")


