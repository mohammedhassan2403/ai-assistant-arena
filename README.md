# AI Assistant Arena

## Overview
This project compares an Open Source AI Assistant and a Frontier Model Assistant.

## Features
- Multi-turn conversations
- Conversational memory
- Open-source LLM support
- Frontier model support
- Hallucination evaluation
- Bias & safety testing
- Gradio UI

## Models Used

### Open Source Model
- Qwen 2.5

### Frontier Model
- GPT-4.1

## Tech Stack
- Python
- Gradio
- Hugging Face
- OpenAI API

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Evaluation
The assistants were evaluated on:
- Hallucination rate
- Bias and harmful outputs
- Content safety
- Jailbreak resistance

## Architecture Decisions
- Gradio used for lightweight UI
- Session-based memory for conversations
- Separate pipelines for OSS and Frontier models

## Tradeoffs
- Smaller OSS models improve latency
- Frontier models provide better reasoning quality

## Future Improvements
- Add vector database memory
- Add observability
- Deploy OSS model publicly
- Add tool calling