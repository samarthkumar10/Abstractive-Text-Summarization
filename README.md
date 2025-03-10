# Abstractive Text Summarizer

## Overview
This project implements an abstractive text summarization system using the T5 Transformer model. It utilizes the Hugging Face `transformers` library for text processing and fine-tuning capabilities. A Streamlit-based UI enables easy user interaction with the summarization model.

## Features
- Implementation of a transformer-based model for text summarization.
- Utilization of a pre-trained `t5-base` model for high-quality abstractive summarization.
- Efficient tokenization and inference using PyTorch.
- Integration with Streamlit for an interactive web-based user interface.
- Model state persistence using `torch.save()` to allow reusability.

## Technical Stack
- **Language**: Python
- **Frameworks**: PyTorch, Hugging Face `transformers`
- **UI**: Streamlit
- **Tokenization**: SentencePiece

## Model Workflow
1. **Preprocessing**: Input text is preprocessed by stripping unnecessary whitespace and concatenating into a format suitable for T5.
2. **Tokenization**: The text is tokenized using the T5 tokenizer with truncation and padding for efficient processing.
3. **Model Inference**: The model generates a summary based on the encoded input representation, with configurable length constraints.
4. **Post-processing**: The generated tokens are decoded into human-readable text.

## Deployment Strategy
- The application can be deployed publicly using Streamlit.
