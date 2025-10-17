import torch
from transformers import pipeline

model = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0 if torch and torch.cuda.is_available() else -1,
)


def predict(prompt):
    input_length = len(prompt.split())
    max_len = min(60, input_length)  # Adjust max_length based on input size

    summary = model(prompt, max_length=max_len, min_length=5, do_sample=False)[0][
        "summary_text"
    ]
    # summary = model(prompt)[0]["summary_text"]
    print("Summary:", summary)
    return summary


predict(
    " THE STRANGE CASE OF DOCTOR JEKYLL AND MR. HYDE Robert Louis Stevenson InfoBooks.org SYNOPSIS OF THE STRANGE CASE OF DR. JEKYLL AND MR. HYDE The Strange Case of Dr. Jekyll and Mr. Hyde is a short psychological horror novel, a true classic of universal literature that deals with a very human and complex theme. Its exquisite descriptions and the atmosphere of mystery that remains until "
)
