from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import math

# ==========================
# 1️⃣ Load dataset
# ==========================
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_texts = dataset['train']
test_texts = dataset['test']

# ==========================
# 2️⃣ Load model và tokenizer
# ==========================
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# ==========================
# 3️⃣ Kiểm tra GPU
# ==========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔍 Using device: {device}")
print(f"💪 GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected'}")

model = model.to(device)

# ==========================
# 4️⃣ Tokenization
# ==========================
def tokenize_function(examples):
    return tokenizer(examples["text"], return_special_tokens_mask=True)

tokenized_train = train_texts.map(tokenize_function, batched=True, remove_columns=["text"])
tokenized_test = test_texts.map(tokenize_function, batched=True, remove_columns=["text"])

# Lọc bỏ mẫu rỗng
tokenized_train = tokenized_train.filter(lambda example: len(example["input_ids"]) > 0)
tokenized_test = tokenized_test.filter(lambda example: len(example["input_ids"]) > 0)

# ==========================
# 5️⃣ Data collator
# ==========================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==========================
# 6️⃣ Training arguments
# ==========================
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    report_to="none"
)

# ==========================
# 7️⃣ Trainer
# ==========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print(f"🧠 Trainer is using device: {trainer.args.device}")

# ==========================
# 8️⃣ Train
# ==========================
trainer.train()

# ==========================
# 9️⃣ Evaluate
# ==========================
eval_results = trainer.evaluate()
print("📊 Evaluation results:", eval_results)
print(f"🤔 Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# ==========================
# 🔟 Generate text
# ==========================
prompt = "Artificial intelligence will change"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=30, do_sample=True)
print("\n📝 Generated text:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
