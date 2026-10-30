
# Day 1 - NLP Overview, Text Pipeline and Tokenization

## Project Objective

This project demonstrates basic NLP text preprocessing and tokenization using:

- Python
- NLTK
- spaCy
- Hugging Face BERT Tokenizer
- Pandas
- Pytest

## Project Features

1. HTML tag removal
2. URL removal
3. Lowercase conversion
4. Punctuation removal
5. Extra whitespace removal
6. NLTK word tokenization
7. NLTK sentence tokenization
8. Stopword removal
9. Negation word preservation
10. spaCy tokenization
11. Hugging Face BERT tokenization
12. Customer review processing pipeline
13. Pytest unit testing

## Installation

Activate the virtual environment and install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download NLTK resources:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

Install spaCy English model:

```powershell
python -m spacy download en_core_web_sm
```

## Run the Project

Run the complete review pipeline:

```powershell
python -m src.main
```

Run spaCy demonstration:

```powershell
python -m src.spacy_demo
```

Run Hugging Face demonstration:

```powershell
python -m src.hf_tokenization
```

Run tests:

```powershell
python -m pytest -v
```

## Tokenization Comparison

### 1. NLTK

NLTK provides word and sentence tokenization.

It is useful for basic NLP preprocessing and traditional NLP tasks.

### 2. spaCy

spaCy provides fast and efficient tokenization.

It is commonly used in production NLP applications.

### 3. Hugging Face BERT Tokenizer

BERT uses subword tokenization.

A word can be divided into smaller subword tokens.

It also generates input IDs and attention masks required by transformer models.

## Conclusion

NLTK, spaCy and Hugging Face use different tokenization approaches.

NLTK is useful for basic NLP tasks, spaCy is useful for efficient NLP processing, and BERT tokenization prepares text for transformer models.
