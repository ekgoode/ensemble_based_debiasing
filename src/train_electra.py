
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def preprocess(example):
    return tokenizer(example['premise'], example['hypothesis'], truncation=True, padding='max_length', max_length=tokenizer.model_max_length)

if __name__ == "__main__":
    
    # Load data
    dataset = load_dataset("snli")

    # Initialize model + tokenizer
    model_name = "google/electra-small-discriminator"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    if hasattr(model, 'electra'):
            for param in model.electra.parameters():
                if not param.is_contiguous():
                    param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


    dataset = dataset.filter(lambda ex: ex['label'] != -1)
    encoded_dataset = dataset.map(preprocess, batched=True)
    encoded_dataset = encoded_dataset.rename_column("label", "labels")
    encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Set training arguments
    training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    )

    # Instnatiate trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
    )
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the trained model and tokenizer
    output_dir = "./models"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
