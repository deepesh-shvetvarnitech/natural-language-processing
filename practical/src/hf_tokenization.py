from transformers import AutoTokenizer


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    text = "I love Natural Language Processing!"

    encoded = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])

    decoded_text = tokenizer.decode(encoded["input_ids"][0])

    print("Original Text:")
    print(text)

    print("\nBERT Tokens:")
    print(tokens)

    print("\nInput IDs:")
    print(encoded["input_ids"])

    print("\nAttention Mask:")
    print(encoded["attention_mask"])

    print("\nDecoded Text:")
    print(decoded_text)


if __name__ == "__main__":
    main()
