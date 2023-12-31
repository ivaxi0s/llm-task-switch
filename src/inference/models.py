from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import openai
from src.tools.tools import get_default_device

HF_MODEL_URLS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "llama-7b": "meta-llama/Llama-2-7b-chat-hf",
}

OPENAI_MODELS = {
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4": "gpt-4",
}


class OpenAIModel:
    """Class wrapper for models that interacts with an API"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = openai

    def predict_batch(self, prompt_batch: list[str]) -> list[str]:
        """Predict a batch of prompts"""
        msgs = [{"role": "user", "content": prompt} for prompt in prompt_batch]
        responses = [
            self.client.ChatCompletion.create(
                model=OPENAI_MODELS[self.model_name], messages=[msg], temperature=0
            )
            for msg in msgs
        ]
        return [r.choices[0].message.content for r in responses]

    def predict_batch_iteratively(self, prompt_batch: list[str]) -> list[str]:
        """Predict a batch of prompts"""
        msgs_batches = []
        for prompts in prompt_batch:
            msgs = []
            for turn in prompts:
                msgs.append({"role": turn["role"], "content": turn["content"]})
            msgs_batches.append(msgs)

        responses = [
            self.client.ChatCompletion.create(
                model=OPENAI_MODELS[self.model_name], messages=msgs, temperature=0
            )
            for msgs in msgs_batches
        ]
        return [r.choices[0].message.content for r in responses]

class HFModel:
    def __init__(self, device, model_name="mistral-7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            HF_MODEL_URLS[model_name], padding_side="left"
        )
        # print(self.tokenizer.padding_side)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(HF_MODEL_URLS[model_name])
        self.model.to(device)
        self.device = device

        # TODO: check if model outputs input prompt tokens or just the ouptut

    def predict_batch(self, prompt_batch: list[str]) -> list[str]:
        """

        # TODO: perhaps all the prompts here should be passed in
        # Then the tokenizer can tokenize in one go?

        # NOT TODO: batching in different order is too cumbersome

        padding: Adds 0s to ensure prompts are the same size
        truncation: Pads upto the largest prompt in the batch
        """
        prompt_batch = [f"[INST]{prompt}[/INST]" for prompt in prompt_batch]

        inputs = self.tokenizer(
            prompt_batch, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        # breakpoint()

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=200,
                temperature=0,
                top_p=1,
            )

        # Remove the tokens that were in the prompt
        output_tokens = output[:, inputs["input_ids"].shape[1] :]
        # Batch decode tokens
        batch_output_text = self.tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )  # NOTE batch decode strips the text by default
        # breakpoint()
        return batch_output_text

    def predict(self, prompt):
        # breakpoint()
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
        breakpoint()
        output_tokens = output_tokens[inputs["input_ids"].shape[1] :]
        # breakpoint()
        output_text = self.tokenizer.decode(
            output_tokens, skip_special_tokens=True
        ).strip()
        return output_text

    def predict_batch_iterative(self, prompt_batch: list[str]) -> list[str]:
        """
            in context examples are passed iteratively
            assume prompt-batch is batch size x number_of_conversation_turns (role specified)
        """
        #TODO from here
        prompt_batch = [f"[INST]{prompt}[/INST]" for prompt in prompt_batch]

        inputs = self.tokenizer(
            prompt_batch, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        # breakpoint()

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=200,
                temperature=0,
                top_p=1,
            )

        # Remove the tokens that were in the prompt
        output_tokens = output[:, inputs["input_ids"].shape[1] :]
        # Batch decode tokens
        batch_output_text = self.tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )  # NOTE batch decode strips the text by default
        # breakpoint()
        return batch_output_text


def get_model(model_name: str, gpu_id: str) -> HFModel | OpenAIModel:
    """Load / intialise the model and return it"""
    if model_name in OPENAI_MODELS.keys():
        model = OpenAIModel(model_name)
    elif model_name in HF_MODEL_URLS.keys():
        model = HFModel(
            device=get_default_device(gpu_id),
            model_name=model_name,
        )
    else:
        raise ValueError(
            f"Unknown model name {model_name}"
            f"Expected model names: {OPENAI_MODELS.keys()} or {HF_MODEL_URLS.keys()}"
        )

    return model
