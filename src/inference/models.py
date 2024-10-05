from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed  # type: ignore
import torch
import openai
from src.tools.tools import get_default_device, DTYPE
from tqdm import tqdm

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

    def predict_batch_iteratively(self, prompt_batch: list[list[dict]]) -> list[str]:
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
        self.model.to(device, dtype=DTYPE)
        self.device = device

        self.max_new_tokens = 512  # Twice the 99.9th percentile of train set summaries
        print(f"Max new tokens: {self.max_new_tokens}")

        self.generation_kwargs = dict(
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            temperature=0,
            top_p=1,
        )

    @torch.no_grad()
    def generate_random_history(self, seed, number=20, max_new_tokens=150) -> list[dict]:
        """Generate a random history: [{user: ..., assistant: ...}, ...]"""

        prompts = []

        set_seed(seed)
        for _ in tqdm(range(number)):
            # 1. Generate a random user message
            output = self.model.generate(
                self.tokenizer("", return_tensors="pt").input_ids.to(self.device),
                do_sample=True,
                temperature=1.0,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            output_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)[
                0
            ]

            # 2. Generate assistant response
            msg = {"role": "user", "content": output_text}
            encodeds = self.tokenizer.apply_chat_template([msg], return_tensors="pt")
            inputs = encodeds.to(self.device)
            output = self.model.generate(
                inputs,
                **self.generation_kwargs | {"max_new_tokens": max_new_tokens},
            )
            system_text = self.tokenizer.batch_decode(output, skip_special_tokens=True)[
                0
            ]
            system_text = system_text.split("[/INST]")[-1].strip()

            prompts.append(
                [
                    {"role": "user", "content": output_text},
                    {"role": "assistant", "content": system_text},
                ]
            )

        return prompts

    @torch.no_grad()
    def predict_batch_iteratively(self, prompt_batch: list[list[dict]]) -> list[str]:
        """
        in context examples are passed iteratively
        assume prompt-batch is batch size x number_of_conversation_turns (role specified)

        NOTE: can only handle a batch size of 1
        """
        if len(prompt_batch) > 1:
            raise ValueError(f"Batch size {prompt_batch} must be 1")

        prompts = prompt_batch[0]
        msgs = []
        for turn in prompts:
            msgs.append({"role": turn["role"], "content": turn["content"]})

        encodeds = self.tokenizer.apply_chat_template(msgs, return_tensors="pt")
        # return [encodeds.shape[-1]] # Debug token length

        inputs = encodeds.to(self.device)

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
            response_probabilities.append(
                probs[response_tokens[0, idx]].float().numpy()
            )
        return response_probabilities


def get_model(model_name: str, gpu_id: int) -> HFModel | OpenAIModel:
    """Load / intialise the model and return it"""
    model: OpenAIModel | HFModel
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
