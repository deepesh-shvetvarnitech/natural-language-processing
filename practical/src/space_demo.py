import spacy


def main():
    nlp = spacy.load("en_core_web_sm")

    text = "This product is excellent! I really love it."

    doc = nlp.make_doc(text)

    tokens = [token.text for token in doc]

    print("Original Text:")
    print(text)

    print("\nspaCy Tokens:")
    print(tokens)


if __name__ == "__main__":
    main()
