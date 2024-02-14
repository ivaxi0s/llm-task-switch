from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import openai
import requests
import json
import time
from src.tools.tools import get_default_device

HF_MODEL_URLS = {
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.1",
    "llama-7b": "meta-llama/Llama-2-7b-chat-hf",
}

OPENAI_MODELS = {
    "gpt3.5": "gpt-3.5-turbo",
    "gpt4": "gpt-4-turbo-preview",
}

BLABLA_MODELS = {"mixtral": "Mixtral-8x7B-Instruct-v0.1"}


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


class BlaBlaModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.url = "https://helmholtz-blablador.fz-juelich.de:8000/v1/chat/completions"
        self.headers = {
            "accept": "application/json",
            "Authorization": "Bearer glpat-4fFjdRCHW98cheMbZt2r",
            "Content-Type": "application/json",
        }

    def predict_batch(self, prompt_batch):
        msgs = [{"role": "user", "content": prompt} for prompt in prompt_batch]
        responses = []
        for msg in msgs:
            data = {
                "model": BLABLA_MODELS[self.model_name],
                "messages": msg,
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "n": 1,
                "max_tokens": 800,  # Adjust max_tokens as needed
                "stop": ["string"],
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "user": "string",
            }

            response = requests.post(
                self.url, headers=self.headers, data=json.dumps(data)
            )
            response_dict = response.json()
            answer = response_dict["choices"][0]["message"]["content"]
            responses.append(answer)

        return responses

    def predict_batch_iteratively(self, prompt_batch):
        msgs_batches = []
        for prompts in prompt_batch:
            msgs = [
                {"role": turn["role"], "content": turn["content"]} for turn in prompts
            ]
            msgs_batches.append(msgs)

        responses = []
        for msgs in msgs_batches:
            data = {
                "model": BLABLA_MODELS[self.model_name],
                "messages": msgs,
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "n": 1,
                "max_tokens": 3000,  # Adjust max_tokens as needed
                "stop": ["string"],
                "stream": False,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "user": "string",
            }

            for _ in range(10):  # Retry up to 5 times
                try:
                    response = requests.post(
                        self.url, headers=self.headers, data=json.dumps(data)
                    )
                    response_dict = (
                        response.json()
                    )  # Raises a JSONDecodeError if the response is empty or not valid JSON
                except json.JSONDecodeError as err:
                    print(f"JSON decoding error occurred: {err}")
                    time.sleep(30)  # Wait for 1 second before retrying
                else:
                    break  # Exit the loop if the request is successful
            else:
                print("Failed to get a successful response after 5 attempts")
                continue  # Skip this iteration and proceed with the next msgs in msgs_batches
            # breakpoint()
            answer = response_dict["choices"][0]["message"]["content"]
            responses.append(answer)
        return responses


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

        self.max_new_tokens = 512  # Twice the 99.9th percentile of train set summaries
        print(f"Max new tokens: {self.max_new_tokens}")

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
                max_new_tokens=self.max_new_tokens,
                temperature=0,
                top_p=1,
            )

        # Remove the tokens that were in the prompt
        output_tokens = output[:, inputs["input_ids"].shape[1] :]
        # Check if max_new_tokens reached
        if output_tokens.shape[1] == self.max_new_tokens:
            num = sum(o != self.tokenizer.eos_token_id for o in output[:, -1])
            print(f"WARNING: max_new_tokens reached; seqs truncated: {num}")

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
                max_new_tokens=self.max_new_tokens,
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

    def predict_batch_iteratively(self, prompt_batch: list[str]) -> list[str]:
        """
        in context examples are passed iteratively
        assume prompt-batch is batch size x number_of_conversation_turns (role specified)

        Unfortunately can only handle a batch size of 1
        """
        if len(prompt_batch) > 1:
            raise ValueError(
                "Batch size cannot be bigger than one for iterative template"
            )

        prompts = prompt_batch[0]
        msgs = []
        for turn in prompts:
            msgs.append({"role": turn["role"], "content": turn["content"]})

        encodeds = self.tokenizer.apply_chat_template(msgs, return_tensors="pt")
        # breakpoint()

        inputs = encodeds.to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                inputs,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                temperature=0,
                top_p=1,
            )

        # Batch decode tokens
        output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)[
            0
        ]  # NOTE batch decode strips the text by default

        # remove input text
        output_text = output_text.split("[/INST]")[-1]
        # breakpoint()
        return [output_text]

    # @torch.no_grad()
    # def predict_batch_auto_iterative(self, prompt_batch: list[list[dict]]) -> list[str]:
    #     """Predict a batch of prompts

    #     Assume
    #         - prompt_batch is {user, system}
    #         - the last turn is the user
    #         - the system responses have been generated previously

    #     """
    #     # breakpoint()
    #     if len(prompt_batch) > 1:
    #         raise ValueError(
    #             "Batch size cannot be bigger than one for iterative template"
    #         )

    #     prompts, _ = prompt_batch
    #     msgs = [{"role": turn["role"], "content": turn["content"]} for turn in prompts]

    #     if len(msgs) < 2:
    #         raise ValueError(
    #             "At least two user turns are required for auto iterative template"
    #         )

    #     # Split the prompts into history and target msgs
    #     *history, target = msgs

    @torch.no_grad()
    def response_probabilities(self, history: list[dict], response: str):
        """Return the likelihood of the response given the history

        Params:
            history: list[(user, model), ...]
            response: expected response

        Returns:
            likelihood: float
        """

        msgs = [{"role": turn["role"], "content": turn["content"]} for turn in history]
        encodeds = self.tokenizer.apply_chat_template(msgs, return_tensors="pt")

        # Tokenize with the response
        # breakpoint()
        msgs.append({"role": "assistant", "content": response})
        response_tokens = self.tokenizer.apply_chat_template(msgs, return_tensors="pt")
        # Extract the response tokens not in the history
        response_tokens = response_tokens[:, encodeds.shape[-1] :]
        # response_tokens = self.tokenizer.encode(response, return_tensors="pt")
        # breakpoint()

        response_probabilities = []
        # Calculate the likelihood of the response
        for idx in range(response_tokens.shape[-1]):
            # Concatenate the history and response_tokens upto idx (excl)
            inputs = torch.cat([encodeds, response_tokens[:, :idx]], dim=1)
            inputs = inputs.to(self.device)
            # Extract logits
            logits = self.model.forward(input_ids=inputs)["logits"]
            probs = torch.softmax(logits[0, -1, :].cpu(), dim=-1)
            # Find the probability of the current response token
            response_probabilities.append(probs[response_tokens[0, idx]].numpy())
        return response_probabilities


def get_model(model_name: str, gpu_id: str) -> HFModel | OpenAIModel | BlaBlaModel:
    """Load / intialise the model and return it"""
    if model_name in OPENAI_MODELS.keys():
        model = OpenAIModel(model_name)
    elif model_name in HF_MODEL_URLS.keys():
        model = HFModel(
            device=get_default_device(gpu_id),
            model_name=model_name,
        )
    elif model_name in BLABLA_MODELS.keys():
        model = BlaBlaModel(model_name)
    else:
        raise ValueError(
            f"Unknown model name {model_name}"
            f"Expected model names: {OPENAI_MODELS.keys()} or {HF_MODEL_URLS.keys()}"
        )

    return model
