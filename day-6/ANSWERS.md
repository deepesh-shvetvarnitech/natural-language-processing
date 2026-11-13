# Day-6 Answers

## 1. What is an LLM?
A Large Language Model is a neural network trained on large amounts of text
to understand and generate language.

## 2. What is a chat-completions message?
It is a list of role/content messages. A `system` message gives behavior
instructions; a `user` message contains the user's request.

## 3. What does temperature do?
Temperature changes the randomness of token selection. Lower values usually
produce more consistent output; higher values usually produce more varied output.

## 4. What does top-p do?
Top-p (nucleus sampling) limits token selection to a probability-ranked group
whose cumulative probability reaches the configured threshold.

## 5. What is a vector database?
A vector database stores embeddings and supports similarity search. It can
retrieve semantically related content even when exact keywords differ.

## 6. What is RAG?
Retrieval-Augmented Generation retrieves relevant external content and provides
it to an LLM as context before the model generates an answer.

## 7. Why validate API input?
Validation prevents invalid parameter values and gives clients clear 422 errors
before an external API request is made.
