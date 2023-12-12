from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_URLS = {"mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1"}


class HFModel:
    def __init__(self, device, model_name="mistral-7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_URLS[model_name])
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_URLS[model_name])
        self.model.to(device)
        self.device = device

    def predict(self, prompt):
        inputs = self.tokenizer(f"[INST]{prompt}[/INST]", return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                # top_k=top_k,
                do_sample=False,
                max_new_tokens=200,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        output_tokens = output[0]
        output_tokens = output_tokens[inputs["input_ids"].shape[1] :]
        # breakpoint()
        output_text = self.tokenizer.decode(
            output_tokens, skip_special_tokens=True
        ).strip()
        return output_text
