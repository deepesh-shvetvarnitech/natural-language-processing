SECTION B: Short Answer – 25 Marks
1. NLP Overview and Real-World Use Cases

Answer:

NLP stands for Natural Language Processing. It is a branch of Artificial Intelligence that helps computers understand, process and analyze human language.

Four real-world examples:
1. Sentiment Analysis: Reviews positive, negative ya neutral hain, ye identify karna.
2. Chatbots: Users ke questions ko understand karke automatic replies dena.
3. Document Search: Documents mein user ke search query se related information find karna.
4. Spam Detection: Emails ko spam aur non-spam categories mein classify karna.
E-commerce example:

E-commerce company NLP ka use customer reviews analyze karne ke liye kar sakti hai. NLP identify kar sakta hai ki customer product se satisfied hai ya dissatisfied.

2. Text Processing Pipeline

Answer:
Raw Text
   ↓
Cleaning
   ↓
Normalization
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Feature Extraction
   ↓
Model


1. Raw Text: Customer se original review receive hota hai.
2. Cleaning: HTML tags, URLs aur extra spaces remove karte hain.
3. Normalization: Text ko lowercase jaise consistent format mein convert karte hain.
4. Tokenization: Text ko individual words ya tokens mein divide karte hain.
5. Stopword Removal: Common words remove karte hain, jab task ke liye suitable ho.
6. Feature Extraction: Text ko numerical features mein convert karte hain.
7. Model: Model review ko positive ya negative classify karta hai.
3. Tokenization Basics

Answer:

Tokenization text ko smaller units called tokens mein divide karne ki process hai.

Word Tokenization: Sentence ko individual words aur punctuation mein divide karta hai.

Sentence Tokenization: Paragraph ko individual sentences mein divide karta hai.

Given sentence:
FastAPI makes AI backend development easier!


Expected word tokens:
["FastAPI", "makes", "AI", "backend", "development", "easier", "!"]



Punctuation ko bhi separate token banaya ja sakta hai. Is example mein exclamation mark ! ek separate token hai.

4. Cleaning vs Tokenization

Answer:

Text Cleaning: Unwanted elements jaise HTML tags, URLs, extra spaces aur unnecessary punctuation remove karna.

Tokenization: Text ko smaller units yani tokens mein divide karna.

Given noisy review:
<p>Great   product!!! Visit <https://example.com> now.</p>


Cleaned version:
great product visit now


Possible tokens:
["great", "product", "visit", "now"]



5. Classical Tokenization vs LLM Tokenization

Answer:

NLTK/spaCy Word Tokenization:
- Text ko words aur punctuation mein divide karta hai.
- Linguistic text processing mein useful hai.
- spaCy POS tagging, NER aur lemmatization jaise features bhi provide karta hai.
Hugging Face Subword Tokenization:
- Text ko words ya smaller subword pieces mein divide karta hai.
- Tokens ko numerical Token IDs mein convert karta hai.
- Transformer models ke liye input prepare karta hai.