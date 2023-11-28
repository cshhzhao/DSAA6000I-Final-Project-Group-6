import os
import operator
import argparse
from typing import Iterator
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import torch
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
import math
import gradio as gr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    tokenizer = LlamaTokenizer.from_pretrained("./checkpoint",
                                            padding_side = 'left',
                                            fast_tokenizer=True, legacy=True)

    return tokenizer

def get_prompt(message: str = "", chat_history: list[tuple[str, str]] = [], system_prompt: str = "") -> str:
    system_prompt= ""
    texts = [f"{system_prompt}\n"]
    for user_input, response in chat_history:
        texts.append(f"{user_input.strip()} {response.strip()}")
    texts.append(f"{message.strip()}")
    return "".join(texts)

def generate_output(
    model,
    tokenizer,
    prompt,
    max_new_tokens = 100,
    temperature = 0.7,
    top_p = 1.0,
    top_k = 1,
    do_sample = True,
    repetition_penalty = 1.5):

  # Due to the prompt structure, we cannot directly truncate the prompt text. See the following operations, when the length of prompt bigger than max length
  max_words = 1024
  promt_words = prompt.split(' ') # according to the blank sign ' ', we can split the prompt text to many words as a list object.
  prompt_words_num = len(promt_words) # statistic the word number.
  if(prompt_words_num >= max_words): # truncation operation
      cut_off_num = prompt_words_num - max_words

      # claim is necessary for fake news detection, but evidence could be cut off to some extendt.
      split_sentence = prompt.split('Evaluate the following assertion:')

      evidence = split_sentence[0]
      evidence_words = evidence.split(' ')
      cut_off_evidence = " ".join(evidence_words[:-1*cut_off_num])

      claim = split_sentence[1]
      prompt = cut_off_evidence + '. ' + 'Evaluate the following assertion:' +  claim

  # change the prompt text to prompt tokens, word id format.
  # inputs = tokenizer(prompt, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
  inputs = tokenizer(prompt, return_tensors="pt").to(device)

  generate_ids = model.generate(inputs.input_ids,
                                attention_mask = inputs.attention_mask,
                                max_new_tokens = max_new_tokens,
                                temperature = temperature,
                                top_p = top_p,
                                top_k = top_k,
                                do_sample = do_sample,
                                num_beams=1,
                                num_beam_groups=1,
                                do_sample=do_sample,
                                num_return_sequences=1,
                                repetition_penalty = repetition_penalty)
  result = tokenizer.batch_decode(generate_ids,
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
  return result



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
        self.tokenizer = tokenizer

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

        device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=device)

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
            max_new_tokens: int = 2000,
            temperature: float = 0.9,
            top_p: float = 1.0,
            top_k: int = 40,
            repetition_penalty: float = 1.5,
            **kwargs: Any,
        ) -> Iterator[str]:
        results = generate_output(self.model,
                                  self.tokenizer,
                                  prompt,
                                  max_new_tokens = max_new_tokens,
                                  temperature = temperature,
                                  top_p = top_p,
                                  top_k = top_k,
                                  do_sample = True,
                                  repetition_penalty = repetition_penalty)
        outputs = []
        for text in results[0]:
          outputs.append(text)
          yield "".join(outputs)

    def run(
        self,
        message: str,
        chat_history: list[tuple[str, str]] = [],
        system_prompt: str = "",
        max_new_tokens: int = 2000,
        temperature: float = 0.9,
        top_p: float = 1.0,
        top_k: int = 40,
        repetition_penalty: float = 1.5,
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
