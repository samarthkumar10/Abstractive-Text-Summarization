import streamlit as st
import torch
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sentencepiece

device = torch.device('cpu')
model_save_path = 'summarization_model.sav'

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base')

    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print("✅ Loaded saved model weights from:", model_save_path)
    else:
        print("⚠️ Saved model not found. Using fresh 't5-base' model.")

    model.to(device)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

st.title("📝 Abstractive Text Summarizer")
st.write("Paste any text below, and I'll summarize it for you using a T5 model.")

text_input = st.text_area("Enter the text to summarize:", height=300)

if st.button("Summarize"):
    if not text_input.strip():
        st.warning("Please enter some text to summarize.")
    else:
        with st.spinner("Summarizing..."):
            preprocessed_text = text_input.strip().replace('\n', ' ')
            t5_input_text = "summarize: " + preprocessed_text

            tokenized_text = tokenizer.encode(
                t5_input_text,
                return_tensors='pt',
                max_length=512,
                truncation=True
            ).to(device)

            summary_ids = model.generate(
                tokenized_text,
                min_length=60,
                max_length=120
            )

            summary = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )

        st.subheader("Summary:")
        st.success(summary)
