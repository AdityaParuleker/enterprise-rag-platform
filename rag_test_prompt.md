# RAG Evaluation Prompt — Nimbus Cloud Storage Knowledge Base

**Purpose:** Use this after ingesting `nimbus_knowledge_base.pdf` into your RAG system. It gives the
system-under-test one instruction block plus a query set with ground-truth answers, so you (or another
LLM acting as a judge) can score retrieval and answer quality automatically.

---

## Instruction block (paste into the RAG system being tested)

```
You are being evaluated as a RAG assistant. You have access to one ingested document:
"Nimbus Cloud Storage Inc. — Internal Knowledge Base" (v4.2).

Answer only from the retrieved context. Rules:
1. If the answer is explicitly stated in the document, give it precisely, including exact
   numbers, dates, and names.
2. If the answer requires combining two or more facts from the document, compute/combine them
   and show which facts you used.
3. If the document does not contain the answer, say clearly that the information is not in the
   knowledge base. Do not guess or use outside knowledge.
4. If a question is ambiguous or underspecified relative to the document, ask which
   interpretation is meant, or answer for each plausible interpretation.
Now answer each of the following questions independently.
```

---

## Query set and expected results

Each row lists: the scenario it stresses, the query, the expected answer (ground truth from the PDF),
and what a pass/fail looks like.

### A. Exact single-fact lookup
1. **Q:** "Who is the CEO of Nimbus?"
   **Expected:** Elena Marsh. **Fail if:** wrong name, or says "not found."
2. **Q:** "What encryption standard does Nimbus use for data at rest?"
   **Expected:** AES-256. **Fail if:** confuses with TLS 1.3 (transit) or omits.
3. **Q:** "How much does the Pro plan cost per month?"
   **Expected:** $29/month. **Fail if:** wrong tier's price returned.

### B. Numeric / table lookup
4. **Q:** "How much storage do I get on the Business plan?"
   **Expected:** 10 TB.
5. **Q:** "What is the API rate limit per minute for the Starter tier?"
   **Expected:** 100 requests per minute.
6. **Q:** "What's the maximum individual file size on the Pro tier?"
   **Expected:** 5 GB. **Fail if:** it returns the 500 GB Business/Enterprise figure instead.

### C. Multi-hop reasoning (requires combining ≥2 facts)
7. **Q:** "How many times bigger is the Business plan's storage than the Pro plan's?"
   **Expected:** 5x (10 TB ÷ 2 TB), and the document states this explicitly as a sanity check.
8. **Q:** "If a Business-tier customer has 90 minutes of downtime in a month beyond the SLA
   threshold, what service credit are they entitled to?"
   **Expected:** 30% of that month's fee (10% per each additional 30 minutes × 3 = 30%), capped
   at 100%. Tests arithmetic + retrieval together.
9. **Q:** "Can a Starter-tier user get a Technical Account Manager, and if not, what's the cheapest
   way to get one?"
   **Expected:** Starter doesn't get one; TAM is a paid Business-tier add-on ($500/mo) or free/included
   at Enterprise — cheapest paid route is the Business add-on.

### D. Negation / absence checks (tests hallucination resistance)
10. **Q:** "Does Nimbus offer an on-premise or self-hosted version of its software?"
    **Expected:** No — explicitly stated Nimbus is hosted-only.
11. **Q:** "Does Nimbus provide 24/7 phone support?"
    **Expected:** No — explicitly stated, email/chat only.
12. **Q:** "Is the Starter plan eligible for SLA service credits?"
    **Expected:** No — only Business and Enterprise are SLA-credit eligible.

### E. Out-of-scope / not-in-document (tests refusal instead of hallucination)
13. **Q:** "What is Nimbus's stock ticker symbol?"
    **Expected:** Not in the document — should say information isn't available, not invent a ticker.
14. **Q:** "How much does the Nimbus mobile app cost separately from a subscription?"
    **Expected:** Not stated anywhere (mobile app is mentioned only for offline sync) — should say not found.
15. **Q:** "What is Nimbus's total annual revenue?"
    **Expected:** Not in the document — should decline/say unknown.

### F. Paraphrase / synonym robustness (same fact, different wording)
16. **Q:** "Who's in charge of engineering at Nimbus?" (paraphrase of "CTO")
    **Expected:** Raj Patel.
17. **Q:** "What's the money-back window if I don't like the service?" (paraphrase of "refund policy")
    **Expected:** 30-day money-back guarantee, plus the prorated-annual-cancellation + $15 fee detail.
18. **Q:** "Where in Europe does Nimbus keep servers?" (paraphrase of "data center locations")
    **Expected:** Frankfurt, Germany (EU-Central).

### G. Entity disambiguation
19. **Q:** "What does 'Nimbus' refer to in this document — is it a company or a product?"
    **Expected:** Both — Nimbus Cloud Storage Inc. is the company, and "Nimbus" is also the name of its
    flagship product; the assistant should note the dual usage rather than pick one arbitrarily.
20. **Q:** "What was introduced in version 2.5 versus version 3.0?"
    **Expected:** v2.5 "Nimbus Share" = shared folders/link sharing; v3.0 "Nimbus AI Search" =
    AI-powered semantic search. Tests that retrieval doesn't merge adjacent table rows.

### H. Chunk-boundary stress test
21. **Q:** "What certifications does Nimbus hold, and in what years were they obtained?"
    **Expected:** SOC 2 Type II (2020) and ISO 27001 (2021) — both in the same paragraph in Section 3;
    tests whether a chunk split mid-paragraph drops one of the two.
22. **Q:** "List every subscription tier and its support response time."
    **Expected:** Starter 48h, Pro 24h, Business 4h, Enterprise 1h — all four rows of the Section 2
    table; tests whether retrieval returns the whole table, not just the top rows.

### I. Ambiguous / underspecified query (tests clarification behavior)
23. **Q:** "What's the storage limit?"
    **Expected:** A good system should ask which tier, or list all four (100GB/2TB/10TB/Unlimited)
    rather than guessing one.
24. **Q:** "When was it released?"
    **Expected:** Ambiguous ("it" = which version/product) — should ask for clarification or list the
    full version history.

### J. Long-tail / rare-term retrieval
25. **Q:** "What is the Private Encryption Key add-on and which version introduced it?"
    **Expected:** Add-on (Business/Enterprise) that prevents Nimbus staff from recovering files if the
    key is lost; introduced in v3.1 "Nimbus Vault," September 2024.
26. **Q:** "What happens if I exceed the API rate limit?"
    **Expected:** HTTP 429 status code returned.

---

## Scoring guide

For each query, score:
- **Retrieval hit (Y/N):** Did the retrieved chunks contain the needed fact(s)?
- **Answer correctness (Y/N):** Does the generated answer match the expected answer above?
- **Hallucination flag (Y/N):** Did the system state anything not supported by the document
  (especially for section E and D)?
- **Refusal correctness (Y/N, section E only):** Did it correctly decline instead of guessing?

Suggested pass threshold: ≥90% correctness on A–D and F–H (core retrieval), 100% correct refusal on
section E (no hallucinated facts), and reasonable clarification behavior on section I.
