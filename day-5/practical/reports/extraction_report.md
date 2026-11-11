# Day 5 Extraction Quality Report

Total reviews processed: 20
Detected brand mentions: 8
Detected product mentions: 8
Detected issue mentions: 1

## Top 10 Keywords

- quality: 5
- samsung: 3
- excellent: 3
- phone: 3
- sound: 3
- speaker: 3
- shoes: 3
- poor: 3
- quickly: 2
- s24: 2

## Top 5 Common Issues

- stopped working: 1

## Five Entity Extraction Samples

### Review: Samsung Galaxy S24 battery drains quickly.
Entities: [{'text': 'Samsung Galaxy S24', 'label': 'ORG', 'start': 0, 'end': 18}, {'text': 'Samsung', 'label': 'BRAND', 'start': 0, 'end': 7}, {'text': 'Galaxy S24', 'label': 'PRODUCT', 'start': 8, 'end': 18}]
Issues: []

### Review: The Samsung camera quality is excellent.
Entities: [{'text': 'Samsung', 'label': 'ORG', 'start': 4, 'end': 11}, {'text': 'Samsung', 'label': 'BRAND', 'start': 4, 'end': 11}]
Issues: []

### Review: My iPhone 15 works perfectly and the battery lasts all day.
Entities: [{'text': '15', 'label': 'CARDINAL', 'start': 10, 'end': 12}, {'text': 'all day', 'label': 'DATE', 'start': 51, 'end': 58}, {'text': 'iPhone 15', 'label': 'PRODUCT', 'start': 3, 'end': 12}]
Issues: []

### Review: OnePlus phone overheats after the latest update.
Entities: [{'text': 'OnePlus', 'label': 'BRAND', 'start': 0, 'end': 7}]
Issues: []

### Review: AirPods Pro sound quality is amazing.
Entities: [{'text': 'AirPods Pro', 'label': 'PRODUCT', 'start': 0, 'end': 11}]
Issues: []


## Five Similar Review Samples

### Query: Samsung Galaxy S24 battery drains quickly.
Results: [{'review_id': 1, 'review': 'Samsung Galaxy S24 battery drains quickly.', 'score': 1.0, 'rank': 1}, {'review_id': 16, 'review': 'My phone battery drains very fast.', 'score': 0.7401, 'rank': 2}, {'review_id': 12, 'review': 'Samsung Galaxy S24 performance is smooth.', 'score': 0.6056, 'rank': 3}]

### Query: The Samsung camera quality is excellent.
Results: [{'review_id': 2, 'review': 'The Samsung camera quality is excellent.', 'score': 1.0, 'rank': 1}, {'review_id': 15, 'review': 'The camera quality is poor but battery life is excellent.', 'score': 0.7632, 'rank': 2}, {'review_id': 10, 'review': 'Sony speaker has excellent sound quality.', 'score': 0.5119, 'rank': 3}]

### Query: My iPhone 15 works perfectly and the battery lasts all day.
Results: [{'review_id': 3, 'review': 'My iPhone 15 works perfectly and the battery lasts all day.', 'score': 1.0, 'rank': 1}, {'review_id': 16, 'review': 'My phone battery drains very fast.', 'score': 0.417, 'rank': 2}, {'review_id': 15, 'review': 'The camera quality is poor but battery life is excellent.', 'score': 0.3844, 'rank': 3}]

### Query: OnePlus phone overheats after the latest update.
Results: [{'review_id': 4, 'review': 'OnePlus phone overheats after the latest update.', 'score': 1.0, 'rank': 1}, {'review_id': 19, 'review': 'The phone gets hot while charging.', 'score': 0.5493, 'rank': 2}, {'review_id': 16, 'review': 'My phone battery drains very fast.', 'score': 0.3204, 'rank': 3}]

### Query: AirPods Pro sound quality is amazing.
Results: [{'review_id': 5, 'review': 'AirPods Pro sound quality is amazing.', 'score': 1.0, 'rank': 1}, {'review_id': 10, 'review': 'Sony speaker has excellent sound quality.', 'score': 0.6046, 'rank': 2}, {'review_id': 2, 'review': 'The Samsung camera quality is excellent.', 'score': 0.4541, 'rank': 3}]


## Analysis

spaCy detects common statistical entities such as organizations, locations, dates, and money when the model recognizes them.

Brand and product names often require custom PhraseMatcher or EntityRuler patterns because the default model may not identify domain-specific names correctly.

Extraction errors can occur when a product name is unknown, an issue is expressed indirectly, or a phrase has different wording.

Extraction can be improved by adding domain terminology, expanding rule patterns, collecting labeled examples, and training a custom NER model.

A product-quality dashboard can group extracted issues by product and brand, count negative mentions, and track issue frequency over time.