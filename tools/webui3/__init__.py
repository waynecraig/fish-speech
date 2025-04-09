from typing import Callable

import gradio as gr

from fish_speech.i18n import i18n
from tools.webui2.variables import HEADER_MD


def build_app(inference_fct: Callable, theme: str = "light") -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Base()) as app:
        gr.Markdown(HEADER_MD)

        # Use light theme by default
        app.load(
            None,
            None,
            js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', '%s');window.location.search = params.toString();}}"
            % theme,
        )

        # Chat interface
        with gr.Row():
            with gr.Column(scale=3):
                audio_input = gr.Audio(
                    label="录音",
                    type="filepath",
                )
                send_button = gr.Button(
                    value="\U0001f4e4 发送",
                    variant="primary",
                )
                error = gr.HTML(
                    label=i18n("Error Message"),
                    visible=True,
                )
                audio_output = gr.Audio(
                    label="生成音频",
                    type="numpy",
                    interactive=False,
                    visible=True,
                )

            with gr.Column(scale=3):
                message_list = gr.Chatbot(type="messages", label="消息列表")

        # Submit
        send_button.click(
            inference_fct,
            [audio_input, message_list],
            [message_list, audio_output, error],
            show_progress="full",
            concurrency_limit=1,
        )

    return app
