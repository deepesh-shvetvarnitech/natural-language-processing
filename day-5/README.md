# Day 5 Review Intelligence Service

A FastAPI microservice that combines sentiment analysis, TF-IDF keyword
extraction, spaCy named entity recognition, custom entity extraction,
issue detection, and semantic review search.

## Features

- Sentiment classification
- TF-IDF keyword extraction
- spaCy NER
- Custom brand, product, and issue extraction
- Embedding-based similar review search
- Combined review analysis endpoint
- Request ID and latency logging

## Setup

powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm