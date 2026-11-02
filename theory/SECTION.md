SECTION A: Objective Answers (20 Marks)
Q. 	     Answer
1		Raw text → cleaning → tokenization → stopword removal → normalization/features
2		High-frequency words with low predictive value
3		A possibly non-dictionary root form
4		A valid base dictionary form of a word
5		studies → studi
6		better → good
7		Grammatical categories such as noun, verb, adjective
8		token.pos_
9		token.lemma_
10		A single token
11		A sequence of two consecutive tokens
12		Unigrams and bigrams
13		Frequent in a document but rare across documents
14		Inverse Document Frequency
15		max_df
16		min_df
17		A sparse matrix
18	   Finding important keywords in customer reviews
19		To apply the same preprocessing logic consistently
20		Save the fitted vectorizer and use the same transformation during inference

                                                    SECTION B
                                                   QUESTION = 1
Pipeline:

Raw Review → Cleaning → Tokenization → Stopword Removal → Stemming/Lemmatization → TF-IDF → ML Model
- Raw Review: Original customer review, for example, “This product is really good!”
- Cleaning: Removes HTML tags, URLs, unwanted characters, and extra spaces.
- Tokenization: Splits the review into individual words or tokens.
- Stopword Removal: Removes common words such as “the”, “is”, and “a” that may have low predictive value.
- Stemming/Lemmatization: Converts words into their root or base forms.
- TF-IDF: Converts text into numerical features so a machine learning model can understand it.
- ML Model: Uses these features to classify the review as positive or negative.
This pipeline helps the product-review classification system process text consistently and identify useful patterns.


                                                   QUESTION = 2
Stemming: Removes word endings using simple rules. It may produce a non-dictionary word.

Lemmatization: Converts a word into its meaningful dictionary base form using linguistic information.
Word	Possible Stemming	Lemmatization
running	run	run
studies	studi	study
better	better	good
mice	mice	mouse
was	wa	be

Note: Outputs depend on the stemming algorithm and lemmatizer.

Which is better for a customer-support chatbot?

Lemmatization is generally better because it produces meaningful base words. This helps the chatbot understand different forms of words more accurately.


                                                       QUESTION = 3
POS tagging means assigning grammatical categories to words, such as noun, verb, adjective, and adverb.

Sentence:

The customer quickly returned the damaged product.
Word	POS Tag	Meaning
The	DET	Determiner
customer	NOUN	Noun
quickly	ADV	Adverb
returned	VERB	Verb
damaged	ADJ	Adjective
product	NOUN	Noun

POS information helps lemmatization identify the correct base form of a word.

For example, the word running may be treated as a verb or noun depending on its context. POS tagging helps the lemmatizer select the appropriate lemma.
                                                     QUESTION = 4
TF (Term Frequency): Measures how frequently a word appears in a document.

IDF (Inverse Document Frequency): Measures how rare or common a word is across all documents.

TF-IDF: Combines TF and IDF to give importance to a word in a document.
\[
TF\text{-}IDF = TF \times IDF
\]

In a dataset containing 100 reviews:
- product may appear in 90 reviews.
- battery life may appear in only 10 reviews.
The word product is common across many reviews, so its IDF value may be low.

The phrase battery life is less common and may provide useful information about a specific product feature, so its IDF value may be higher.

Therefore, battery life can receive a higher TF-IDF score than product, depending on its frequency within the individual review.
                                                      QUESTION =5
Unigram: A single token.

Example: battery

Bigram: Two consecutive tokens.

Example: battery life

Trigram: Three consecutive tokens.

Example: battery life excellent

Review:

The battery life is excellent but the camera quality is poor.

Useful bigrams:
1. battery life
2. life is
3. is excellent
4. camera quality
5. quality is
6. is poor
Why are bigrams useful?

Bigrams capture relationships between consecutive words. They help identify product features and sentiment expressions.

