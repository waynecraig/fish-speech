import html
from functools import partial
import json
from typing import Any, Callable
import uuid
import os

from dashscope.audio.asr import Transcription
from gradio import ChatMessage
from openai import OpenAI
import requests
from fish_speech.i18n import i18n
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
from http import HTTPStatus


def inference_wrapper(audio_input, message_list, engine):
    """
    Wrapper for the inference function.
    Handles audio-to-text, text-to-text, and text-to-audio transformations.
    """

    # Step 1: Convert input audio to text
    audio_text = audio_to_text(audio_input)
    message_list.append({"role":"user", "content":audio_text})

    # Step 2: Generate response text
    response_text = generate_response_text(message_list, audio_text)
    message_list.append({"role":"assistant", "content":response_text})

    # Step 3: Convert response text to audio
    audio_output, error = text_to_audio(response_text, engine)

    return message_list, audio_output, error


def audio_to_text(audio_input: Any) -> str:
    """
    Convert audio input to text using DashScope paraformer-v2 model.
    """
    filename = uuid.uuid4().hex
    newfilepath = f"{os.environ.get('AUDIO_DIR')}/{filename}.wav"

    # copy the audio file to a new location
    with open(audio_input, "rb") as f:
        with open(newfilepath, "wb") as newf:
            newf.write(f.read())
    
    fileurl = f"{os.environ.get('AUDIO_URL')}/{filename}.wav"

    task_response = Transcription.async_call(
        model='paraformer-v2',
        file_urls=[fileurl],
        language_hints=['zh']
    )
    transcribe_response = Transcription.wait(task=task_response.output.task_id)
    if transcribe_response.status_code == HTTPStatus.OK:
        print(transcribe_response.output)
        transcription_url = transcribe_response.output["results"][0]["transcription_url"]
        response = requests.get(transcription_url)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        result = response.json()
        return result["transcripts"][0]["text"]
    else:
        raise Exception(f"Transcription failed: {transcribe_response.status_code} - {transcribe_response.message}")


def generate_response_text(message_list, input_text: str) -> str:
    """
    Generate response text using DashScope qwen-plus model.
    """
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    print("message_list:", message_list)
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": msg["role"], "content": msg["content"]} for msg in message_list]
    )
    return completion.choices[0].message.content or ""


def text_to_audio(text: str, engine) -> tuple:
    """
    Convert text to audio using the local TTS model.
    """
    print("Text to audio:", text)
    req = ServeTTSRequest(
        text=text,
        reference_id=None,
        references=get_reference_audio(
            reference_audio="S0278.wav",
            reference_text="我认为品冠唱的最后才学会蛮好听，播一首",
        ),
        max_new_tokens=0,
        chunk_length=200,
        top_p=0.7,
        repetition_penalty=1.2,
        temperature=0.7,
        seed=None,
        use_memory_cache="on",
    )

    for result in engine.inference(req):
        match result.code:
            case "final":
                return result.audio, None
            case "error":
                return None, build_html_error_message(i18n(result.error))
            case _:
                pass

    return None, i18n("No audio generated")


def get_reference_audio(reference_audio: str, reference_text: str) -> list:
    """
    Get the reference audio bytes.
    """

    with open(reference_audio, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]


def build_html_error_message(error: Any) -> str:
    """
    Build an HTML error message.
    """
    error = error if isinstance(error, Exception) else Exception("Unknown error")

    return f"""
    <div style="color: red; 
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


def get_inference_wrapper(engine) -> Callable:
    """
    Get the inference function with the immutable arguments.
    """
    return partial(
        inference_wrapper,
        engine=engine,
    )
