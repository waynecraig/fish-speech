from typing import Callable

import gradio as gr

from fish_speech.i18n import i18n
from tools.webui2.variables import HEADER_MD, TEXTBOX_PLACEHOLDER


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

        # Inference
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label=i18n("Input Text"), placeholder=TEXTBOX_PLACEHOLDER, lines=10
                )

            with gr.Column(scale=3):
                with gr.Row():
                    error = gr.HTML(
                        label=i18n("Error Message"),
                        visible=True,
                    )
                with gr.Row():
                    audio = gr.Audio(
                        label=i18n("Generated Audio"),
                        type="numpy",
                        interactive=False,
                        visible=True,
                    )

                with gr.Row():
                    with gr.Column(scale=3):
                        generate = gr.Button(
                            value="\U0001f3a7 " + i18n("Generate"),
                            variant="primary",
                        )

        # Submit
        generate.click(
            inference_fct,
            [
                text,
                # Default values for removed advanced settings
                gr.State(""), # reference_id
                gr.State("ygsdeq_2.wav"), # reference_audio
                gr.State("影照看起来也睡着了，他起伏的胸口表示他还活着在，灵魂尚且有机会回归他的身体之中, 他感到一阵欣慰的暖流通过，好像还冒得么事可怕的事情发生在他身上。"), # reference_text
                gr.State(0),  # max_new_tokens
                gr.State(200),  # chunk_length
                gr.State(0.7),  # top_p
                gr.State(1.2),  # repetition_penalty
                gr.State(0.7),  # temperature
                gr.State(0),  # seed
                gr.State("on"),  # use_memory_cache
            ],
            [audio, error],
            concurrency_limit=1,
        )

    return app
