                                           SECTION = A
Q.No.		Answer
1	       A numerical vector representing text meaning
2		   Individual words or subwords
3		   The meaning of a full sentence or short text
4		   all-MiniLM-L6-v2
5		   384
6		   Angle-based similarity between two vectors
7		   1
8		   -1
9		  Cosine similarity
10		  The most similar vectors to a query vector
11		  cosine_distances
12	      Lexical/word-overlap patterns
13		  Query and document use different words with similar meaning
14		 “cheap phone with good camera” for “affordable mobile with excellent camera”
15		 [0.12, -0.44, 0.91, ...]
17		 TF-IDF vector with mostly zero values
18		 The same embedding model and vector space
19		 Finding similar customer complaints
20		 Dot product on normalized vectors behaves like cosine similarity

                                                 SECTION B
                                                QUESTION=1
Word Embeddings
Word embeddings represent individual words or subwords as numerical vectors. Each word is mapped into a vector space where words with similar meanings may have similar representations.

For example:
The phone battery is excellent and lasts two days.


In a word embedding approach, each word is represented separately:
The       → [0.12, 0.45, ...]
phone     → [0.21, 0.67, ...]
battery   → [0.33, 0.54, ...]
excellent → [0.78, 0.92, ...]


The sentence is represented through individual word vectors.
Sentence Embeddings
Sentence embeddings represent the meaning of an entire sentence or short text as a single numerical vector.

For example:
The phone battery is excellent and lasts two days.


The complete sentence is converted into one vector:
Sentence → [0.21, -0.45, 0.67, ..., 0.12]




Q2. Cosine Similarity Intuition (5 Marks)
Definition
Cosine similarity measures the similarity between two vectors by calculating the cosine of the angle between them.

It focuses on the direction of vectors rather than their magnitude.
- If the angle is 0°, cosine similarity is 1.
- If the angle is 90°, cosine similarity is 0.
- If the angle is 180°, cosine similarity is -1.
Formula
\[
\text{Cosine Similarity}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

Where:
- \(A\cdot B\) = Dot product of two vectors.
- \(\|A\|\) = Magnitude of vector A.
- \(\|B\|\) = Magnitude of vector B.
Given
\[
A=[1,2,2]
\]

\[
B=[2,4,4]
\]

Step 1: Calculate the dot product
\[
A\cdot B=(1\times2)+(2\times4)+(2\times4)
\]

\[
=2+8+8=18
\]

Step 2: Calculate the magnitude of A
\[
\|A\|=\sqrt{1^2+2^2+2^2}
\]

\[
=\sqrt{1+4+4}=\sqrt9=3
\]

Step 3: Calculate the magnitude of B
\[
\|B\|=\sqrt{2^2+4^2+4^2}
\]

\[
=\sqrt{4+16+16}=\sqrt{36}=6
\]

Step 4: Calculate cosine similarity
\[
\text{Cosine Similarity}=\frac{18}{3\times6}
\]

\[
=\frac{18}{18}
\]

\[
\boxed{\text{Cosine Similarity}=1.0}
\]


                                       QUESTION = 3
Aspect	TF-IDF	Embeddings
Vector type	Sparse vector	Dense vector
Captures synonyms?	Limited ability	Better ability
Needs trained model?	No deep learning model required	Yes, typically a pretrained model
Best use case	Keyword-based search	Semantic search
Example strength	Exact word matching	Meaning-based matching

TF-IDF Similarity
TF-IDF calculates the importance of words in documents.

It mainly captures lexical similarity, meaning that it works well when the query and document contain similar words.

Example:
Query: Excellent camera quality
Review: Excellent camera quality


TF-IDF will likely produce high similarity because the words overlap.
Embedding Similarity
Embedding similarity converts text into dense numerical vectors that represent semantic meaning.

It can identify similar meanings even when different words are used.

Example:
Query: Cheap phone with good camera

Review: Affordable mobile with excellent camera


The words "cheap" and "affordable" have similar meanings. Similarly, "phone" and "mobile" are related.

Embedding-based semantic search can identify this relationship better than traditional TF-IDF.
Example Where TF-IDF Is Better
Query: Error code 404
Document: Error code 404 occurred


TF-IDF is useful when exact keyword matching is important.

                                           QUESTION = 4
Definition
Nearest-neighbor search is a technique used to find the most similar vectors to a given query vector.

In semantic search, text is converted into numerical embeddings. The system compares the query embedding with stored review embeddings and retrieves the closest matches.
Working with 100 Review Embeddings
Suppose we have 100 product reviews.
Review 1  → Embedding 1
Review 2  → Embedding 2
Review 3  → Embedding 3
...
Review 100 → Embedding 100


Query:
My phone battery drains quickly.


The search process works as follows:
1. Convert the query into an embedding using the same model.
2. Compare the query embedding with all 100 review embeddings.
3. Calculate cosine distance or cosine similarity.
4. Sort the reviews according to similarity.
5. Select the top 5 nearest reviews.
6. Return the review text, review ID, score, and rank.
Example
Review	Similarity Score	Rank
Battery drains very fast	0.94	1
Phone loses charge quickly	0.91	2
Battery performance is poor	0.87	3
Mobile battery is weak	0.83	4
Charging does not last long	0.80	5

Similarity Score vs Distance Score
Similarity Score	Distance Score
Higher means more similar	Lower means more similar
Measures closeness in meaning	Measures separation between vectors
Cosine similarity of identical vectors = 1	Cosine distance of identical vectors = 0

For cosine distance:
\[
\text{Cosine Distance}=1-\text{Cosine Similarity}
\]

For example:
\[
1-0.94=0.06
\]

Therefore, a similarity score of 0.94 corresponds to a cosine distance of 0.06.

Conclusion: Nearest-neighbor search efficiently retrieves the most relevant reviews by comparing the query embedding with stored embeddings and selecting the top-k closest vectors.

                                          QUESTION = 5
A model produces 384-dimensional embeddings.
1. How many floating-point values are stored for one review?
A 384-dimensional embedding contains 384 floating-point values.
\[
\boxed{384\text{ values}}
\]

2. How many values are stored for 100 reviews?
\[
384\times100=38,400
\]

\[
\boxed{38,400\text{ values}}
\]

3. Why can higher dimensions improve meaning representation but increase memory and computation?
Higher-dimensional embeddings can represent more complex semantic relationships and capture additional information about text.

However, they also have disadvantages:
- They require more memory to store.
- They increase computational requirements.
- Similarity calculations become more expensive.
- Searching through large embedding collections can become slower.
For example, a 768-dimensional embedding requires twice as many values as a 384-dimensional embedding.
4. What is truncate_dim and when might it be useful?
truncate_dim is a parameter in the Sentence Transformer API that allows us to reduce the dimensionality of the output embedding.

For example:
from sentence_transformers import SentenceTransformermodel = SentenceTransformer(    "sentence-transformers/all-MiniLM-L6-v2")embedding = model.encode(    "The phone battery is excellent.",    truncate_dim=128)print(embedding.shape)



Output:
(128,)


Normally, the model produces 384-dimensional embeddings. By using truncate_dim=128, the output is reduced to 128 dimensions.

It can be useful when:
- Reducing storage requirements.
- Improving search speed.
- Reducing computational cost.
- Working with memory-constrained systems.
Conclusion: Embedding dimensions determine the number of values used to represent text. Higher dimensions may provide richer representations, while lower dimensions can reduce memory and computation.