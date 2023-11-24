from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    default_data_collator,
    get_scheduler,
    AutoConfig,
)
from transformers.integrations import HfDeepSpeedConfig
from threading import Thread
from typing import Any, Iterator, Union, List
import torch
import math

def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    tokenizer = LlamaTokenizer.from_pretrained("./checkpoint",
                                            padding_side = 'left',
                                            fast_tokenizer=True, legacy=True)

    return tokenizer

def get_prompt(
    message: str, chat_history: list[tuple[str, str]] = [], system_prompt: str = ""
) -> str:
        """Process message to llama2 prompt with chat history
        and system_prompt for chatbot.

        Examples:
            >>> prompt = get_prompt("Hi do you know Pytorch?")

        Args:
            message: The origianl chat message to generate text from.
            chat_history: Chat history list from chatbot.
            system_prompt: System prompt for chatbot.

        Yields:
            prompt string.
        """
        texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
        for user_input, response in chat_history:
            texts.append(f"{user_input.strip()} [/INST] {response.strip()} </s><s> [INST] ")
        texts.append(f"{message.strip()} [/INST]")
        return "".join(texts)


class llama_wrapper:
    def __init__(self,
                model_class,
                model_name_or_path,
                tokenizer,
                ds_config=None,
                rlhf_training=False,
                dropout=None,
                bf16 = False):
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        self.configure_dropout(model_config, dropout)
        self.tokenizer = tokenizer.to(device="cuda")

        # Note: dschf is defined in function scope to avoid global effects
        # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            dschf = HfDeepSpeedConfig(ds_config)
        else:
            dschf = None
        if rlhf_training:
            # the weight loading is handled by create critic model
            model = model_class.from_config(model_config)
        else:
            if not bf16:
                model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config)
            else:
                model = model_class.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=model_config,
                torch_dtype=torch.bfloat16)  

        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model.resize_token_embeddings(int(
            8 *
            math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
        
        self.model = model.to(device="cuda")   

    def configure_dropout(self, model_config, dropout):
        if dropout is not None:
            for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                        'activation_dropout'):
                if hasattr(model_config, key):
                    print(f"Setting model_config.{key} to {dropout}")
                    setattr(model_config, key, dropout)

    def get_token_length(
        self,
        prompt: str,
    ) -> int:
        input_ids = self.tokenizer([prompt], return_tensors="np")["input_ids"]
        return input_ids.shape[-1]

    def get_input_token_length(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
        file: bool = False
    ) -> int:
        if not file:
            prompt = get_prompt(message, chat_history, system_prompt)
        else:
            prompt = get_prompt(message=message, chat_history=[], system_prompt=system_prompt)

        return self.get_token_length(prompt)

    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 1000,
            temperature: float = 0.9,
            top_p: float = 1.0,
            top_k: int = 40,
            repetition_penalty: float = 1.0,
            **kwargs: Any,
        ) -> Iterator[str]:
            """Create a generator of response from a prompt.

            Examples:
                >>> llama2_wrapper = LLAMA2_WRAPPER()
                >>> prompt = get_prompt("Hi do you know Pytorch?")
                >>> for response in llama2_wrapper.generate(prompt):
                ...     print(response)

            Args:
                prompt: The prompt to generate text from.
                max_new_tokens: The maximum number of tokens to generate.
                temperature: The temperature to use for sampling.
                top_p: The top-p value to use for sampling.
                top_k: The top-k value to use for sampling.
                repetition_penalty: The penalty to apply to repeated tokens.
                kwargs: all other arguments.

            Yields:
                The generated text.
            """
            from transformers import TextIteratorStreamer

            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")

            streamer = TextIteratorStreamer(
                self.tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
            )
            generate_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
            generate_kwargs = (
                generate_kwargs if kwargs is None else {**generate_kwargs, **kwargs}
            )
            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()

            outputs = []
            for text in streamer:
                outputs.append(text)
                yield "".join(outputs)

    def run(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
        max_new_tokens: int = 1000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.0,
        file: bool = False
    ) -> Iterator[str]:
        """Create a generator of response from a chat message.
        Process message to llama2 prompt with chat history
        and system_prompt for chatbot.

        Args:
            message: The origianl chat message to generate text from.
            chat_history: Chat history list from chatbot.
            system_prompt: System prompt for chatbot.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            top_k: The top-k value to use for sampling.
            repetition_penalty: The penalty to apply to repeated tokens.
            kwargs: all other arguments.

        Yields:
            The generated text.
        """
        if not file:
            prompt = get_prompt(message, chat_history, system_prompt) 
        else:
            prompt = get_prompt(message=message, chat_history=[], system_prompt=system_prompt)
        return self.generate(
            prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty
        )

