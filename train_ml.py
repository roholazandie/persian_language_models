from tokenizers.implementations import ByteLevelBPETokenizer, BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

tokenizer = BertWordPieceTokenizer(vocab_file="./FaBerto/vocab.txt")

# tokenizer._tokenizer.post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# )
tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("مرکزگرایی یکی از عوامل اصلی خداناباوری است.").tokens)
print(tokenizer.encode("این یک تست است.").ids)



config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)


tokenizer = RobertaTokenizerFast.from_pretrained("./FaBerto", max_len=512)


model = RobertaForMaskedLM(config=config)

print(model.num_parameters())




dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/train.txt",
    block_size=128,
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


training_args = TrainingArguments(
    output_dir="./FaBerto",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

#trainer.train()