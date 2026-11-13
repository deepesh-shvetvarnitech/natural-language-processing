# Vector Database Design: 10,000 Product FAQs

## 1. Goal
Build semantic FAQ retrieval for an ecommerce catalogue containing approximately 10,000 product FAQs. A customer can ask a question in their own words and retrieve the most relevant approved answer.

## 2. Proposed design

| Setting | Decision | Reason |
|---|---|---|
| Collection / index | `product_faqs_v1` | Separate, versioned collection for product FAQ content |
| Vector database | Chroma (persistent client for prototype; Chroma server for shared deployment) | Simple metadata filtering and a practical Python integration |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight general-purpose sentence embeddings |
| Embedding dimension | 384 | Output dimension of the selected model |
| Similarity metric | Cosine distance/similarity | Compares semantic direction of normalized vectors |
| Top-k | 5 | Small context set for the first retrieval pass |

Use the exact same embedding model and preprocessing for indexing and query-time embedding. If the model changes, create a new versioned collection and re-embed all FAQ records.

## 3. Metadata fields

| Field | Example | Purpose |
|---|---|---|
| `faq_id` | `FAQ-000123` | Stable unique FAQ identifier |
| `product_id` | `PHONE-15-128` | Product-level filtering |
| `product_name` | `Example Phone 15` | Human-readable product name |
| `brand` | `ExampleBrand` | Brand filter |
| `category` | `smartphones` | Product category filter |
| `question` | `How long does the battery last?` | Original FAQ question |
| `answer` | `Battery life depends on usage...` | Approved answer text |
| `locale` | `en-IN` | Language/region filter |
| `is_active` | `true` | Exclude retired FAQs |
| `updated_at` | `2026-10-01` | Freshness and maintenance |

Store the embedding with the FAQ question (or question plus a short answer summary) and keep the complete approved answer in the document/metadata payload. Ensure metadata types are supported by the chosen Chroma version.

## 4. Indexing and retrieval flow
1. Validate and de-duplicate the 10,000 FAQ records by `faq_id`.
2. Normalize whitespace and retain product names, model numbers, and technical terms.
3. Embed each FAQ question using `all-MiniLM-L6-v2` (384 dimensions).
4. Upsert the vector, FAQ text, and metadata into `product_faqs_v1`.
5. Embed the incoming customer question with the same model.
6. Apply metadata filters (for example, `is_active=true`, locale, product ID, or category).
7. Search by cosine similarity and retrieve the top 5 candidates.
8. Optionally rerank candidates, then pass the approved FAQ answer(s) to the LLM as grounded context.
9. Return the answer with FAQ IDs so the source can be audited.

## 5. Filtering strategy
- Always filter `is_active=true`.
- Apply `locale` based on the user's selected language/region.
- If the user is viewing a specific product page, filter by `product_id` first.
- Otherwise use extracted brand/category filters only when confidence is sufficient.
- Do not filter on a guessed product ID; fall back to broader semantic search.
- Enforce user/tenant access restrictions in the application before returning results.

## 6. Example query and response

**Customer query:** “Does this phone last a full day without charging?”

**Example search filter:** `is_active=true`, `locale="en-IN"`, `product_id="PHONE-15-128"` (only when the customer is on that product page).

**Illustrative retrieved result (not a measured database result):**

```json
{
  "faq_id": "FAQ-000123",
  "question": "How long does the battery last?",
  "answer": "Battery life varies with screen brightness, network use, and apps. See the product specifications for the rated test conditions.",
  "similarity": 0.82
}
```

The similarity number above is illustrative, not a benchmark. Production thresholds must be calibrated on a labelled FAQ query set. If no candidate clears the threshold, ask a clarifying question or use the normal support flow rather than inventing an answer.

## 7. Evaluation and production considerations
- Create a held-out set of real customer questions with relevant FAQ IDs.
- Measure Recall@5, MRR@5, and no-result rate before release.
- Monitor retrieval latency, embedding errors, stale FAQs, and low-confidence matches.
- Back up the persistent collection and test restore procedures.
- Keep secrets out of source control; secure the database and restrict administrative access.
- For 10,000 FAQs, a local/persistent Chroma deployment is sufficient for a prototype. Reassess scaling, high availability, backup, and operational requirements before production.
