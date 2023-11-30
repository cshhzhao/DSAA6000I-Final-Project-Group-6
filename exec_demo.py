import os
import operator
import argparse
from typing import Iterator
import requests
import re
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
from googleapiclient.discovery import build
import requests

def extract_last_num(text: str) -> float:
    response = text.split('Response')[-1]
    res = re.findall(r"(True|False|true|false)", response, re.IGNORECASE)
    if len(res) > 0:
        return res[-1]
    else:
        return "Sorry, I don't know."

    # response = text.split('Response')
    # res = re.findall(r"(True|False)", text)
    # if len(res) > 0:
    #     return res[-1]
    # else:
    #     return "Sorry, I don't know."

def google_search(query):
    api_key = "AIzaSyDYgTehiaaRT0U8LVJnVEHtHFXMo08aeK8"

    cse_id = "92b1ded210d294914"
    
    query = query

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'key': api_key,
        'cx': cse_id
    }

    result = requests.get(search_url, params=params)

    data = result.json()
    Snippets = ""    
    if 'items' in data:
        for item in data['items']:
            snippet = item.get('snippet')    
            Snippets = Snippets + snippet

    print(Snippets)
    return "###Google Search Result###" + Snippets

if __name__=='__main__':
    print(os.getcwd())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
        tokenizer = LlamaTokenizer.from_pretrained("./checkpoint",
                                                padding_side = 'left',
                                                fast_tokenizer=True, legacy=True)

        return tokenizer
    
    # 设置个功能，决定，用户自己提供evidence还是调用google搜索
    # 如果用户在calim:xxxx. 后面加一句###Google Search On###就是使用谷歌搜索的结果作为evidence，否则用户需要自己提供evidence
    def get_prompt(message: str = "", chat_history: list[tuple[str, str]] = [], system_prompt: str = "") -> str:

        system_prompt = message
            
        return system_prompt
    
    # 设置个功能，决定，用户自己提供evidence还是调用google搜索
    def get_inference_prompt(message: str = "", chat_history: list[tuple[str, str]] = [], system_prompt: str = "") -> str:
        if("###Google Search Off###" in message):
            print(message)
            # 保证输入是 evidence：xxxx. claim: xxxx. 就可以切分出咱们要的内容
            actual_message = message.split('###Google Search Off###')[0]
            split_object = actual_message.split('claim')
            claim = split_object[-1]
            evidence = split_object[0].split('evidence')[-1]
            system_prompt = f"Below is an instruction that describes a fake news detection task." \
                            f" Write a response that appropriately completes the request. ### Instruction: " \
                            f"If there are only True and False categories, based on your knowledge and the following information {evidence}" \
                            f"Evaluate the following assertion {claim}" \
                            f"If possible, please also give the reasons. ### Response:."

            print(system_prompt)
        else:
            print(message)
            # 保证输入是 evidence：xxxx. claim: xxxx. 就可以切分出咱们要的内容
            actual_message, google_search_result = message.split('###Google Search On###')
            split_object = actual_message.split('claim')
            claim = split_object[-1]
            evidence = google_search_result.split("###Google Search Result###")[-1]
            system_prompt = f"Below is an instruction that describes a fake news detection task." \
                            f" Write a response that appropriately completes the request. ### Instruction: " \
                            f"If there are only True and False categories, based on your knowledge and the following information {evidence}" \
                            f"Evaluate the following assertion {claim}" \
                            f"If possible, please also give the reasons. ### Response:."

            print(system_prompt)            

        return system_prompt    

    def generate_output(
        model,
        tokenizer,
        prompt,
        max_new_tokens = 1000,
        temperature = 0.7,
        top_p = 1.0,
        top_k = 1,
        do_sample = False,
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
                max_new_tokens: int = 1000,
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
            #   yield "".join(outputs)

            # 定制化输出内容
            detection_result = extract_last_num(results[0])
            print('预测结果：', results[0])
            yield "We think that the claim is " + detection_result
            # yield "We think that the claim is " + detection_result + "Original Output: " + "".join(outputs)

        def run(
            self,
            message: str,
            chat_history: list[tuple[str, str]] = [],
            system_prompt: str = "",
            max_new_tokens: int = 1000,
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
                prompt = get_inference_prompt(message, chat_history, system_prompt)
            else:
                prompt = get_inference_prompt(message=message, chat_history=[], system_prompt=system_prompt)
            return self.generate(
                prompt, max_new_tokens, temperature, top_p, top_k, repetition_penalty
            )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = load_hf_tokenizer("./checkpoint", fast_tokenizer=True)
    tokenizer.pad_token=tokenizer.eos_token
    model = llama_wrapper(model_class=AutoModelForCausalLM, model_name_or_path="./checkpoint", tokenizer=tokenizer, bf16=True)
    model.model.to(device=device)


    def render_html(text: list[tuple[str, str]]):
        '''
        For chatbot output
        '''
        target_string = text[-1][1]
        if "True" in target_string:
            lowest_index = target_string.find("True")
            up_index = lowest_index + len("True")
            text[-1][1] = f"{target_string[:lowest_index]}<span style='background-color: yellow;'>True</span>{target_string[up_index:]}"
        elif "False" in target_string:
            lowest_index = target_string.find("False")
            up_index = lowest_index + len("False")
            text[-1][1] = f"{target_string[:lowest_index]}<span style='background-color: yellow;'>False</span>{target_string[up_index:]}"

        return text

    def render_text(text: str):
        '''
        For text
        '''
        target_string = text
        if "True" in target_string:
            lowest_index = target_string.find("True")
            up_index = lowest_index + len("True")
            text = f"{target_string[:lowest_index]}<span style='background-color: yellow;'>True</span>{target_string[up_index:]}"
        elif "False" in target_string:
            lowest_index = target_string.find("False")
            up_index = lowest_index + len("False")
            text = f"{target_string[:lowest_index]}<span style='background-color: yellow;'>False</span>{target_string[up_index:]}"
        return text

    def scrape(url):
        driver = webdriver.Chrome()
        driver.get(url=url)
        webpage_content = driver.find_element(by=By.TAG_NAME, value="body").text
        return webpage_content

    def load_file(filepath):
        if os.path.exists(filepath):
            with open(file=filepath, mode="r", encoding="utf-8") as file:
                text = file.read()
            return text
        else:
            raise FileNotFoundError("File not exists")

    def clear_and_save_textbox(message: str) -> tuple[str, str]:
        return "", message

    def display_input(
        message: str, history: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        history.append((message, ""))
        return history

    def delete_prev_fn(
        history: list[tuple[str, str]]
    ) -> tuple[list[tuple[str, str]], str]:
        try:
            message, _ = history.pop()
        except IndexError:
            message = ""
        return history, message or ""

    def check_input_token_length(
    message: str, chat_history: list[tuple[str, str]], system_prompt: str
    ) -> None:
        input_token_length = model.get_input_token_length(message=message, chat_history=chat_history, system_prompt=system_prompt, file=False)
        if input_token_length > 1024:
            raise gr.Error(
                f"The accumulated input is too long ({input_token_length} > {1024}). Clear your chat history and try again."
            )

    def check_file_input_token_length(
    message: str, system_prompt: str
    ) -> None:
        input_token_length = model.get_input_token_length(message=message, system_prompt=system_prompt, file=True)
        if input_token_length > 1024:
            raise gr.Error(
                f"The accumulated input is too long ({input_token_length} > {1024}). Clear your chat history and try again."
            )

    def generate(
            message: str,
            scrape_content: str,
            history_with_input: list[tuple[str, str]],
            system_prompt: str,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
    ) -> Iterator[list[tuple[str, str]]]:
        print("Generate function called with message:", message)  # 打印传入的消息
        if max_new_tokens > 10000:
            raise ValueError
        print(scrape_content)
        message += scrape_content
        history = history_with_input[:-1]
        generator = model.run(
            message,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            file=True
        )
        try:
            first_response = next(generator)
            print("First response from model:", first_response)  # 打印模型的第一个响应
            yield history + [(message, first_response)]
        except StopIteration:
            print("Model did not return any response")  # 打印模型没有返回任何响应的情况
            yield history + [(message, "")]
        for response in generator:
            print("Next response from model:", response)  # 打印模型的后续响应
            yield history + [(message, response)]

    def file_url_generate(
            message: str,
            system_prompt: str,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            top_k: int,
    ) -> Iterator[list[tuple[str, str]]]:
        if max_new_tokens > 20000:
            raise ValueError

        generator = model.run(
            message,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature = temperature,
            top_p=top_p,
            top_k=top_k,
            file=True
        )
        try:
            first_response = next(generator)
            yield [(message, first_response)]
        except StopIteration:
            yield [(message, "")]
        for response in generator:
            yield [(message, response)]


    def two_columns_list(tab_data, chatbot):
            result = []
            for i in range(int(len(tab_data) / 2) + 1):
                row = gr.Row()
                with row:
                    for j in range(2):
                        index = 2 * i + j
                        if index >= len(tab_data):
                            break
                        item = tab_data[index]
                        with gr.Group():
                            gr.HTML(
                                f'<p style="color: black; font-weight: bold;">{item["act"]}</p>'
                            )
                            prompt_text = gr.Button(
                                label="",
                                value=f"{item['summary']}",
                                size="sm",
                                elem_classes="text-left-aligned",
                            )
                            prompt_text.click(
                                fn=clear_and_save_textbox,
                                inputs=prompt_text,
                                outputs=saved_input,
                                api_name=False,
                                queue=True,
                            ).then(
                                fn=display_input,
                                inputs=[saved_input, chatbot],
                                outputs=chatbot,
                                api_name=False,
                                queue=True,
                            ).then(
                                fn=lambda : None,
                                inputs=[saved_input, chatbot, system_prompt],
                                api_name=False,
                                queue=False,
                            ).success(
                                fn=lambda : None,
                                inputs=[
                                    saved_input,
                                    chatbot,
                                    system_prompt,
                                    max_new_tokens,
                                    temperature,
                                    top_p,
                                    top_k,
                                ],
                                outputs=chatbot,
                                api_name=False,
                            )
                    result.append(row)
            return result

    CSS = """
        .contain { display: flex; flex-direction: column;}
        #component-0 #component-1 #component-2 #component-4 #component-5 { height:71vh !important; }
        #component-0 #component-1 #component-24 > div:nth-child(2) { height:80vh !important; overflow-y:auto }
        .text-left-aligned {text-align: left !important; font-size: 16px;}
        .md.svelte-r3x3aw.chatbot {background-color: yellow;}
    """


    prompts = {}
    with gr.Blocks(css=CSS) as demo:
        with gr.Tab("Text"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    gr.Markdown(" ")
                    with gr.Group():
                        chatbot = gr.Chatbot(label="Chatbot", elem_classes="chatbot")
                        with gr.Row():
                            textbox = gr.Textbox(
                                container=False,
                                show_label=False,
                                placeholder="Type a message...",
                                lines=5,
                                scale=12,
                            )
                            submit_button = gr.Button(
                                "Submit", variant="primary", scale=1, min_width=0
                            )
                    with gr.Row():
                        retry_button = gr.Button("🔄  Retry", variant="secondary")
                        undo_button = gr.Button("↩️ Undo", variant="secondary")
                        clear_button = gr.Button("🗑️  Clear", variant="secondary")

                    saved_input = gr.State()
                    scrape_content = gr.State()
                    with gr.Row():
                        advanced_checkbox = gr.Checkbox(
                            label="Advanced",
                            value="",
                            container=False,
                            elem_classes="min_check",
                        )
                        prompts_checkbox = gr.Checkbox(
                            label="Prompts",
                            value="",
                            container=False,
                            elem_classes="min_check",

                        )

                    with gr.Column(visible=True) as advanced_column:
                        system_prompt = gr.Textbox(
                            label="System prompt", value="", lines=6
                        )
                        max_new_tokens = gr.Slider(
                            label="Max new tokens",
                            minimum=1,
                            maximum=1024,
                            step=1,
                            value=512,
                        )
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.1,
                            maximum=4.0,
                            step=0.1,
                            value=1.0,
                        )
                        top_p = gr.Slider(
                            label="Top-p (nucleus sampling)",
                            minimum=0.05,
                            maximum=1.0,
                            step=0.05,
                            value=0.95,
                        )
                        top_k = gr.Slider(
                            label="Top-k",
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=20,
                        )
            textbox.submit(
                fn=clear_and_save_textbox,
                inputs=textbox,
                outputs=[textbox, saved_input],
                api_name=False,
                queue=False,
            ).then(
                fn=google_search,
                inputs=saved_input,
                outputs=scrape_content                
            ).then(
                fn=display_input,
                inputs=[saved_input, chatbot],
                outputs=chatbot,
                api_name=False,
                queue=False,
            ).then(
                fn=check_input_token_length,
                inputs=[saved_input, chatbot, system_prompt],
                api_name=False,
                queue=False,
            ).success(
                fn=generate,
                inputs=[
                    saved_input,
                    scrape_content,
                    chatbot,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                ],
                outputs=chatbot,
                api_name=False,
            ).then(
                fn=render_html,
                inputs=chatbot,
                outputs=chatbot,
                api_name=False
            )

            submit_button.click(
                    fn=clear_and_save_textbox,
                    inputs=textbox,
                    outputs=[textbox, saved_input],
                    api_name=False,
                    queue=False,
                ).then(
                    fn=google_search,
                    inputs=saved_input,
                     outputs=scrape_content,
                    api_name=False,
                    queue=False,                    
                ).then(
                    fn=display_input,
                    inputs=[saved_input, chatbot],
                    outputs=chatbot,
                    api_name=False,
                    queue=False,
                ).then(
                    fn=check_input_token_length,
                    inputs=[saved_input, chatbot, system_prompt],
                    api_name=False,
                    queue=False,
                ).success(
                    fn=generate,
                    inputs=[
                        saved_input,
                        scrape_content,
                        chatbot,
                        system_prompt,
                        max_new_tokens,
                        temperature,
                        top_p,
                        top_k,
                    ],
                    outputs=chatbot,
                    api_name=False,
                ).then(
                    fn=render_html,
                    inputs=chatbot,
                    outputs=chatbot,
                    api_name=False
                )

            retry_button.click(
                fn=delete_prev_fn,
                inputs=chatbot,
                outputs=[chatbot, saved_input],
                api_name=False,
                queue=False,
            ).then(
                fn=display_input,
                inputs=[saved_input, chatbot],
                outputs=chatbot,
                api_name=False,
                queue=False,
            ).then(
                fn=generate,
                inputs=[
                    saved_input,
                    chatbot,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                ],
                outputs=chatbot,
                api_name=False,
            ).then(
                fn=render_html,
                inputs=chatbot,
                outputs=chatbot,
                api_name=False
            )

            undo_button.click(
                fn=delete_prev_fn,
                inputs=chatbot,
                outputs=[chatbot, saved_input],
                api_name=False,
                queue=False,
            ).then(
                fn=lambda x: x,
                inputs=[saved_input],
                outputs=textbox,
                api_name=False,
                queue=False,
            )

            clear_button.click(
                fn=lambda: ([], ""),
                outputs=[chatbot, saved_input],
                queue=False,
                api_name=False,
            )

        with gr.Tab("File"):
            saved_input = gr.State()
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload News File", file_types=[".docx", ".pdf", ".md"], type="filepath", scale=2)
                    submit_file = gr.Button("Detect New")
                    """
                    system_prompt = gr.Textbox(
                        label="System prompt", value="", lines=6
                    )
                    max_new_tokens = gr.Slider(
                        label="Max new tokens",
                        minimum=1,
                        maximum=2048,
                        step=1,
                        value=2048,
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=4.0,
                        step=0.1,
                        value=1.0,
                    )
                    top_p = gr.Slider(
                        label="Top-p (nucleus sampling)",
                        minimum=0.05,
                        maximum=1.0,
                        step=0.05,
                        value=0.95,
                    )
                    top_k = gr.Slider(
                        label="Top-k",
                        minimum=1,
                        maximum=1000,
                        step=1,
                    )
    """
                with gr.Column():
                    output = gr.HTML(label="Output")

            submit_file.click(
                fn=load_file,
                inputs=file_input,
                outputs=saved_input,
                api_name=False
            ).then(
                fn=check_file_input_token_length,
                inputs=[saved_input, system_prompt],
                api_name=False,
                queue=False,
            ).success(
                fn=file_url_generate,
                inputs=[
                    saved_input,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                ],
                outputs=output,
                api_name=False,
            ).then(
                fn=render_text,
                inputs=output,
                outputs=output,
                api_name=False
            )

        with gr.Tab("URL"):
            url_input = gr.Textbox(label="News Url")
            saved_input = gr.State()
            submit_url = gr.Button("Detect News")
            output = gr.Text(label="Output")

            submit_url.click(
                fn=scrape,
                inputs=url_input,
                outputs=saved_input,
                api_name=False
            ).then(
                fn=check_file_input_token_length,
                inputs=[saved_input, system_prompt],
                api_name=False,
                queue=False
            ).then(
                fn=render_text,
                inputs=saved_input,
                outputs=output,
                api_name=False
            ).success(
                fn=file_url_generate,
                inputs=[
                    saved_input,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    top_k,
                ],
                outputs=output,
                api_name=False)

    demo.queue(max_size=20).launch(
        show_api=False,
        share=True,
        ssl_verify=False,
        max_threads=20,
    )
