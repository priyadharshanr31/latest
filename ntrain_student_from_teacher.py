import torch
import pandas as pd
from transformers import BartTokenizerFast, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# ✅ Load teacher predictions
data = torch.load("C:\\CS Tech\\Agents\\Daily\\NEW\\new_teacher_predictions.pth")  # Make sure path is correct

# ✅ Convert to HuggingFace Dataset
df = pd.DataFrame(data)
df['input_text'] = "question: " + df['question'] + " context: " + df['context']
df['output_text'] = df['teacher_answer']
dataset = Dataset.from_pandas(df[['input_text', 'output_text']])

# ✅ Load existing student model
model_path = "C:\\CS Tech\\Agents\\Daily\\NEW\\updated_bart_student_model"
tokenizer = BartTokenizerFast.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

# ✅ Tokenization function
def tokenize(batch):
    inputs = tokenizer(batch["input_text"], padding="max_length", truncation=True, max_length=512)
    targets = tokenizer(batch["output_text"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

# ✅ Tokenize the dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# ✅ Training config
training_args = TrainingArguments(
    output_dir="./student_model_logs",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    evaluation_strategy="no"
)

# ✅ Setup trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# ✅ Train the student model
trainer.train()

# ✅ Save the updated model to a new folder (IMPORTANT to avoid file lock issue)
save_path = "C:\\CS Tech\\Agents\\Daily\\NEW\\updated_bart_student_model_v2"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Student model saved to: {save_path}")
