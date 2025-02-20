from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch

class TestModel:
    def __init__(self, base_hf_model_name, finetuned_hf_model_name):
        login(token='hf_XXXX')
        torch.cuda.empty_cache()
        self.config = PeftConfig.from_pretrained(finetuned_hf_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_hf_model_name)
        self.model = PeftModel.from_pretrained(self.base_model, finetuned_hf_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(base_hf_model_name)
        self.pad_tokenizer()
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def pad_tokenizer(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_text(self, prompt, max_length=100):
        torch.cuda.empty_cache()
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # Move inputs to the same device as the model
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

