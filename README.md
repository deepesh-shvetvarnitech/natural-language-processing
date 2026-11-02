Is API ka main aim hai text ko step-by-step process karna, text se features nikalna, aur text ko numerical form (TF-IDF) mein convert karna, taaki Machine Learning model text par kaam kar sake.

1. GET / — Home Endpoint

Isko kyun banaya?
Ye API ka home ya health-check endpoint hai. Isse pata chalta hai:
- API run ho rahi hai ya nahi.
- API ka title/message kya hai.
- Swagger docs ka path kya hai.
- Kaun-kaun se endpoints available hain.


2. POST /clean — Text Cleaning

Is endpoint ko kyun banaya?
User ka raw text aksar unnecessary cheezon ke saath aata hai, jaise:
- Uppercase/lowercase inconsistency
- HTML tags
- URLs
- Extra punctuation
- Extra spaces
Text cleaning ka kaam raw text ko simple aur consistent banana hai.

{
  "text": "I LOVE NLP!!! Visit https://example.com <b>Welcome</b>"
}

3. POST /tokenize — Tokenization + Stopword Removal

Is endpoint ko kyun banaya?
Machine Learning ko raw sentence ke bajaye words/tokens ki list dena useful hota hai.

Tokenization sentence ko chhote parts mein divide karti hai.

{
  "text": "I LOVE NLP! This is a wonderful movie, and I am enjoying it."
}


4. POST /stemming


Is endpoint ko kyun banaya?
Is endpoint ka purpose hai compare karna ki words ko short/base form mein laane ke do methods kaise kaam karte hain.
- Stemming: Word ke ending ko rules se cut karta hai.
- Lemmatization: Linguistic analysis ke through word ka base form nikalta hai.

{
  "tokens": [
    "playing",
    "played",
    "plays",
    "studies",
    "studying",
    "better",
    "running"
  ]
}



5. POST /pos-tagging — Part of Speech Tagging
Is endpoint ko kyun banaya?
POS Tagging sentence ke har word ka grammatical role identify karti hai.


{
  "text": "The beautiful girl is reading a book."
}

6. POST /preprocess — Complete Text Preprocessing
Is endpoint ko kyun banaya?
Ye ek combined endpoint hai. Ismein tum individual processing steps ka output ek saath dekh sakte ho.

Tumhare code ke according ye:
1. Text clean karta hai.
2. Tokens banata hai.
3. Stopwords remove karta hai.
4. Lemmatization karta hai.
5. Final processed text return karta hai.

{
  "text": "I LOVED the movies!!! They were running <b>beautifully</b> yesterday. Visit https://example.com"
}

7. POST /tfidf-conversion — Text ko Numerical Features mein Convert Karna
Is endpoint ko kyun banaya?
Machine Learning models raw text par directly kaam nahi karte. Text ko numerical representation mein convert karna padta hai.

{
  "texts": [
    "I love this movie. The acting was wonderful.",
    "I love this movie. The story was wonderful.",
    "This movie was boring and the acting was bad.",
    "The story was boring. I did not like this movie.",
    "I love the acting and the story."
  ]
}


8. POST /features — Unigram, Bigram aur Combined Feature Report
Is endpoint ko kyun banaya?
Is endpoint ka purpose hai different n-gram configurations ko compare karna.
- Unigram: One word.
- Bigram: Two consecutive words.
- Unigram + Bigram: One-word aur two-word features.

{
  "texts": [
    "I love this movie. The acting was wonderful.",
    "I love this movie. The story was wonderful.",
    "This movie was boring and the acting was bad.",
    "The story was boring. I did not like this movie.",
    "I love the acting and the story."
  ]
}
