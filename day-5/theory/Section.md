SECTION A: Objective Answers 
Q. 		Answer
1		Named Entity Recognition
2		Named spans such as people, organizations, products, dates, and locations
3	    doc.ents
4		ent.text
5		ent.label_
6		ORG
7		GPE
8		PhraseMatcher
9		Matching large lists of exact multi-word phrases
10		Add custom rule-based entities to doc.ents
11		"battery drains quickly"
12		Important terms based on frequency and rarity
13		Semantic search with embeddings
14		Compare direction/similarity of embedding vectors
15		The same model for query and documents
16		/similar
17		/entities
18		/keywords
19		Load once at application startup
20		Sentiment, confidence, keywords, entities, issues, and similar reviews

SECTION B: Short Answer (25 Marks)

                                            QUESTION = 1
NER stands for Named Entity Recognition.

NER is an NLP technique used to identify and classify important named entities in text, such as people, organizations, products, dates, locations, and monetary values.

It helps an NLP system understand which real-world entities are mentioned in a sentence.

Given review:
I bought a Samsung Galaxy S24 in Mumbai for ₹74,999, but the battery drains quickly.



Identified entities:

Entity	 Likely Label	Explanation
Samsung	  ORG	        Company or brand
Galaxy    S24	        PRODUCT	Product name
Mumbai	  GPE	        Geopolitical entity (city)
₹74,999	  MONEY	        Monetary value








                                                QUESTION = 2


Statistical NER and rule-based extraction are two approaches used to identify entities in text.

1. Statistical NER

Statistical NER uses a trained machine learning model to identify entities based on patterns learned from training data.

Example:
import spacynlp = spacy.load("en_core_web_sm")text = "Samsung launched a new phone in Mumbai."doc = nlp(text)for ent in doc.ents:    print(ent.text, ent.label_)



It can identify organizations, locations, people, dates, and other standard entity types.

2. Rule-Based Extraction

Rule-based extraction uses predefined rules, patterns, or lists to identify specific entities.

Two useful spaCy components are:
- EntityRuler: Adds custom entities to spaCy's entity recognition pipeline using patterns.
- PhraseMatcher: Matches predefined words or multi-word phrases in text.
Example:



Difference:
Statistical NER	                     Rule-Based Extraction
Uses a trained model	             Uses predefined rules and patterns
Identifies standard entities	    Identifies custom entities
Can recognize unseen entities	    Usually matches known patterns
May make prediction errors	        Depends on rule coverage

USES:-
- Use statistical NER for identifying standard entities such as organizations, locations, and dates.
- Use PhraseMatcher or EntityRuler for extracting predefined brands, product names, and product issues.
Conclusion: Combining both approaches provides better entity extraction for product reviews.

                                             QUESTION = 3

TF-IDF stands for Term Frequency–Inverse Document Frequency.

TF-IDF is a statistical technique used to identify important words or phrases in a document.

It calculates importance using two components:
1. Term Frequency (TF): Measures how frequently a term appears in a document.
2. Inverse Document Frequency (IDF): Measures how rare or informative a term is across the collection of documents.
The TF-IDF score becomes higher when a term is frequent in a particular document but relatively rare across the entire collection.

Given review:
The camera quality is poor but battery life is excellent.


Likely important keywords:
- camera
- camera quality
- poor
- battery
- battery life
- excellent
Why are camera quality and battery life important?
- camera quality identifies the specific feature being discussed.
- battery life identifies another important product feature.
- poor indicates a negative opinion.
- excellent indicates a positive opinion.
- Common words such as the, is, and but generally provide less useful information and are often removed as stopwords.
TF-IDF can extract important terms and help identify the main topics discussed in reviews.





Conclusion: TF-IDF helps extract meaningful keywords from reviews by considering term frequency and rarity.

                                                    QUESTION = 4
Embeddings are numerical representations of text. They convert words, sentences, or complete reviews into numerical vectors that capture semantic meaning.

The /similar endpoint uses embeddings and cosine similarity to find reviews with similar meanings.

Input review:
My phone battery drains very fast.


Existing review:
Battery backup is poor and needs charging twice a day.


Although the words are different, both reviews express a similar meaning: the phone has a poor battery performance.

How does semantic search work?
1. Convert existing reviews into embedding vectors using a sentence embedding model.
2. Convert the user's query into an embedding using the same model.
3. Calculate cosine similarity between the query vector and stored review vectors.
4. Rank reviews by similarity score.
5. Return the top-k most similar reviews.

Cosine similarity:
FORMULA LIKHTE NHI BAAN RHA 