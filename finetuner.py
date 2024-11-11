from transformers import AutoTokenizer, AutoModelForCausalLM,Trainer, TrainingArguments, DataCollatorForSeq2Seq
from huggingface_hub import login
from torch.utils.data import Dataset
import torch
import os

os.environ["WANDB_DISABLED"] = "true"

hf_read_token='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'


# predefined class for custom dataset creation for finetuning
class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.labels[idx]
        }


class finetuner:
  def __init__(self, model_name):
    login(token=hf_read_token)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(model_name)

  def enable_gradient_checkpointing(self):
    self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing

  def pad_tokenizer(self):
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

  def tokenize_data(self):
    max_length = 50
    # Tokenize prompts and responses
    self.tokenized_inputs = self.tokenizer(self.inputs, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    self.tokenized_labels = self.tokenizer(self.outputs, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
    # Ensure labels' padding tokens are ignored in loss computation
    self.tokenized_labels[self.tokenized_labels == self.tokenizer.pad_token_id] = -100

  def split_dataset(self):

        # ✂️ Split dataset into training and evaluation sets
        from sklearn.model_selection import train_test_split

        indices = list(range(len(self.tokenized_inputs["input_ids"])))
        train_indices, val_indices = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_seed
        )

        # 🏋️ Training dataset
        train_inputs = {
            "input_ids": self.tokenized_inputs["input_ids"][train_indices],
            "attention_mask": self.tokenized_inputs["attention_mask"][train_indices],
        }
        train_labels = self.tokenized_labels[train_indices]
        self.train_dataset = CustomDataset(train_inputs, train_labels)

        # 🧪 Evaluation dataset
        val_inputs = {
            "input_ids": self.tokenized_inputs["input_ids"][val_indices],
            "attention_mask": self.tokenized_inputs["attention_mask"][val_indices],
        }
        val_labels = self.tokenized_labels[val_indices]
        self.eval_dataset = CustomDataset(val_inputs, val_labels)


  def prepare_dataset(self,inputs,outputs):
    self.inputs=inputs
    self.outputs=outputs
  
  def collate_data(self):
    self.data_collator = DataCollatorForSeq2Seq(
      tokenizer=self.tokenizer,
      model=self.model,
      padding=True)
  
  def prepare_training_Args(self,output_path):
    self.training_args = TrainingArguments(
              output_dir=output_path,
              overwrite_output_dir=True,
              eval_strategy="no",
              learning_rate=2e-5,
              per_device_train_batch_size=1,  # Reduced batch size
              gradient_accumulation_steps=4,  # Gradient accumulation
              num_train_epochs=3,
              weight_decay=0.01,
              fp16=True,  # Enable mixed-precision training
              gradient_checkpointing=True,  # Enable gradient checkpointing
          )
    
    self.trainer = Trainer(
                          model=self.model,
                          args=self.training_args,
                          train_dataset=self.train_dataset,
                          eval_dataset=self.eval_dataset,
                          data_collator=self.data_collator)
    
  def train(self):
    torch.cuda.empty_cache()
    try:
        self.trainer.train()
    except ValueError as e:
        print("\nError during training:")
        print(e)

  def save_model(self):
    self.model.save_pretrained(self.output_path)
    self.tokenizer.save_pretrained(self.output_path)

  def run(self,inputs,outputs,output_path,train_size=0.8,random_seed=42):
    self.random_seed=42
    self.test_size=1-train_size
    self.output_path=output_path
    self.enable_gradient_checkpointing()
    self.pad_tokenizer()	
    self.prepare_dataset(inputs,outputs)
    self.tokenize_data()
    self.prepared_dataset = CustomDataset(self.tokenized_inputs, self.tokenized_labels)
    self.collate_data()
    self.split_dataset()
    self.prepare_training_Args(output_path)
    self.train()
    self.save_model()
