from __future__ import annotations
import re
from html import unescape
from functools import lru_cache
import spacy, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

URL_RE = re.compile(r'(?i)\b(?:https?://|www\.)\S+')
HTML_RE = re.compile(r'<[^>]*>')
BAD_RE = re.compile(r"[^\w\s'’-]", flags=re.UNICODE)
SPACE_RE = re.compile(r'\s+')

@lru_cache(maxsize=1)
def get_nlp():
    try: return spacy.load('en_core_web_sm')
    except OSError as exc:
        raise RuntimeError('Install model: python -m spacy download en_core_web_sm') from exc

def ensure_stopwords():
    try: stopwords.words('english')
    except LookupError: nltk.download('stopwords', quiet=True)

def clean_text(text: str, lowercase: bool=True, remove_html: bool=True, remove_urls: bool=True) -> str:
    if not isinstance(text, str): return ''
    text = unescape(text)
    if remove_html: text = HTML_RE.sub(' ', text)
    if remove_urls: text = URL_RE.sub(' ', text)
    if lowercase: text = text.lower()
    return SPACE_RE.sub(' ', BAD_RE.sub(' ', text)).strip()

def tokenize(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip(): return []
    return [t.text for t in get_nlp().make_doc(text) if not t.is_space]

def remove_stopwords(tokens: list[str]) -> list[str]:
    ensure_stopwords(); stops = set(stopwords.words('english')) - {'no','not','nor'}
    return [t for t in tokens if t.lower() not in stops]

def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer(); return [stemmer.stem(t) for t in tokens]

def lemmatize_tokens(tokens: list[str]) -> list[str]:
    if not tokens: return []
    return [t.lemma_ for t in get_nlp()(' '.join(tokens)) if not t.is_space]

def get_pos_tags(text: str) -> list[tuple[str,str]]:
    if not isinstance(text,str) or not text.strip(): return []
    return [(t.text,t.pos_) for t in get_nlp()(text) if not t.is_space]

def preprocess(text: str, lowercase: bool=True, remove_html: bool=True, remove_urls: bool=True,
               remove_stopwords_flag: bool=True, use_stemming: bool=False,
               use_lemmatization: bool=True) -> str:
    if use_stemming and use_lemmatization: raise ValueError('Choose stemming OR lemmatization, not both.')
    cleaned = clean_text(text,lowercase,remove_html,remove_urls)
    doc = get_nlp()(cleaned)
    ensure_stopwords(); stops = set(stopwords.words('english')) - {'no','not','nor'}
    result=[]
    for tok in doc:
        if tok.is_space or tok.is_punct: continue
        if remove_stopwords_flag and tok.text.lower() in stops: continue
        value = tok.text
        if use_stemming: value = PorterStemmer().stem(value)
        elif use_lemmatization: value = tok.lemma_
        result.append(value.lower() if lowercase else value)
    return ' '.join(result)