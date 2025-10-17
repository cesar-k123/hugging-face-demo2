import torch
from transformers import pipeline
import gradio as gr


model = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if torch and torch.cuda.is_available() else -1,
)


def predict(prompt):
    if not prompt.strip():
        return "Please enter some text to summarize."
    try:
        summary = model(prompt)[0]["summary_text"]
        print("Summary:", summary)
        return summary
    except Exception as e:
        print("Error during summarization:", e)
        return "An error occurred. Please check your input or try again."




with gr.Blocks() as demo:
    gr.Markdown("# Text Summarization with BART")
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to summarize",
                lines=10,
            )
        with gr.Column():
            summary = gr.Textbox(
                label="Summary",
                placeholder="Summary will appear here",
                lines=10,
            )
    btn = gr.Button("Summarize")
    btn.click(fn=predict, inputs=txt, outputs=summary)
demo.launch()