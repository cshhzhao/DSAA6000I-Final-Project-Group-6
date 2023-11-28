import os
import argparse
from typing import Iterator
from app_llama import llama_wrapper, load_hf_tokenizer
from transformers import AutoModelForCausalLM, LlamaTokenizer
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import torch

import gradio as gr

def main():
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

    demo.queue(max_size=20).launch(share=False)


if __name__ == "__main__":
    main()
