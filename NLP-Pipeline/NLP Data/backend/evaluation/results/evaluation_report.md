# Semantic Search Model Evaluation Report

## 1. Overview

This report evaluates the **University Semantic Search System** retrieval quality by running
natural-language queries through `SemanticSearchEngine.search` and comparing ranked **documents**
(PDF filenames) to hand-labeled relevant files in `evaluation/dataset.json`.

Chunk-level hits are collapsed to **document-level** rankings: the first time a filename appears
in the result list defines its rank. This matches how users perceive “which thesis was found.”

## 2. Dataset

- **Source:** `evaluation/dataset.json`
- **Unit of relevance:** PDF `filename` (same field as in `cache/documents.json`).
- **Queries:** short, realistic academic search phrases aligned with titles/topics in the cached corpus.

## 3. Metrics (what, why, when)

| Metric | What it measures | Why it fits semantic search | When it is most useful |
|--------|------------------|----------------------------|-------------------------|
| **Precision@K** | (# relevant in top *K*) / *K* (TREC-style) | Penalizes irrelevant items in fixed *K* slots; with **one** labeled relevant doc at rank 1, P@K = 1/*K* | Fixed cutoff *K* and pooled evaluation |
| **Hit Rate@K** | 1 if any relevant appears in top *K*, else 0 (then averaged) | Simple “success within *K*” for single-target queries | When each query has one must-find document |
| **Recall@K** | Share of all labeled relevant docs found in top-*K* | Shows whether multiple correct theses appear within *K* | Multi-document ground truth, discovery tasks |
| **F1@K** | Harmonic mean of Precision@K and Recall@K | One score balancing accuracy vs. coverage in the top *K* | When both precision and recall in top-*K* matter |
| **MRR** | Average \(1/\text{rank}\) of the **first** relevant hit | Rewards putting any correct answer near the top | “Find one good document” behavior |
| **nDCG@10** | Discounted gain with ideal normalization | Rewards **ordering**: better docs higher; uses rank discount | When rank within the first positions matters |

## 4. Run configuration

- **Retrieval depth (chunks):** 30
- **K for P/R/F1:** 5, 10
- **Vector backend:** FAISS (local)
- **Indexed chunks (engine stats):** 207
- **Queries evaluated:** 6

## 5. Aggregate results

| Metric | Value |
|--------|-------|
| MRR | 1.0000 |
| Mean nDCG@10 | 1.0000 |
| Mean Precision@5 | 0.2000 |
| Mean Recall@5 | 1.0000 |
| Mean F1@5 | 0.3333 |
| Mean Hit Rate@5 | 1.0000 |
| Mean Precision@10 | 0.1000 |
| Mean Recall@10 | 1.0000 |
| Mean F1@10 | 0.1818 |
| Mean Hit Rate@10 | 1.0000 |

## 6. Per-query summary

| ID | P@5 | R@5 | F1@5 | Hit@5 | P@10 | R@10 | F1@10 | Hit@10 | MRR | nDCG@10 |
|----|-----|-----|------|-------|------|------|-------|--------|-----|---------|
| q1 | 0.2000 | 1.0000 | 0.3333 | 1.0000 | 0.1000 | 1.0000 | 0.1818 | 1.0000 | 1.0000 | 1.0000 |
| q2 | 0.2000 | 1.0000 | 0.3333 | 1.0000 | 0.1000 | 1.0000 | 0.1818 | 1.0000 | 1.0000 | 1.0000 |
| q3 | 0.2000 | 1.0000 | 0.3333 | 1.0000 | 0.1000 | 1.0000 | 0.1818 | 1.0000 | 1.0000 | 1.0000 |
| q4 | 0.2000 | 1.0000 | 0.3333 | 1.0000 | 0.1000 | 1.0000 | 0.1818 | 1.0000 | 1.0000 | 1.0000 |
| q5 | 0.2000 | 1.0000 | 0.3333 | 1.0000 | 0.1000 | 1.0000 | 0.1818 | 1.0000 | 1.0000 | 1.0000 |
| q6 | 0.2000 | 1.0000 | 0.3333 | 1.0000 | 0.1000 | 1.0000 | 0.1818 | 1.0000 | 1.0000 | 1.0000 |

## 7. Interpretation

- **High MRR / high P@K:** the embedding model and index place the correct thesis (filename)
  near the top for topic-aligned queries.
- **Low recall@10 with single relevant doc:** the right document is often missing from the
  first 10 *chunks* after deduplication—check indexing, chunking, or query wording.
- **Low nDCG@10 with good MRR:** relevant doc appears but below less relevant neighbors;
  ranking order may need re-ranking or a different embedding model.
- **Low mean Precision@K but Hit Rate@K = 1:** common when each query has a single relevant
  label: TREC-style P@K divides by *K*, so a lone correct hit at rank 1 yields P@K = 1/*K*.
- **Zero metrics:** engine not initialized, empty index, or filenames in the dataset do not
  match `documents.json` / cache (typo check).

## 8. Raw ranked document lists

### q1: local revenue collection and service delivery in local government Tororo municip…

- **Success:** True  **Method:** semantic_faiss
- **Ranked filenames:** `['Chemutai_G_BBA_2025.pdf']`
- **Relevant (ground truth):** `['Chemutai_G_BBA_2025.pdf']`

### q2: ICT services and financial performance of commercial banks Centenary Bank Mukono

- **Success:** True  **Method:** semantic_faiss
- **Ranked filenames:** `['Okwir_D_BBA_2024.pdf']`
- **Relevant (ground truth):** `['Okwir_D_BBA_2024.pdf']`

### q3: church intervention separated parents children Paidha Parish Nebbi Diocese Ugand…

- **Success:** True  **Method:** semantic_faiss
- **Ranked filenames:** `['Cwinyaai_W_BDIV_2025.pdf']`
- **Relevant (ground truth):** `['Cwinyaai_W_BDIV_2025.pdf']`

### q4: revenue mobilization correlation service delivery municipal local government

- **Success:** True  **Method:** semantic_faiss
- **Ranked filenames:** `['Chemutai_G_BBA_2025.pdf']`
- **Relevant (ground truth):** `['Chemutai_G_BBA_2025.pdf']`

### q5: theology divinity dissertation Church of Uganda bishop school

- **Success:** True  **Method:** semantic_faiss
- **Ranked filenames:** `['Cwinyaai_W_BDIV_2025.pdf', 'Okwir_D_BBA_2024.pdf', 'Chemutai_G_BBA_2025.pdf']`
- **Relevant (ground truth):** `['Cwinyaai_W_BDIV_2025.pdf']`

### q6: digital banking technology and bank performance case study

- **Success:** True  **Method:** semantic_faiss
- **Ranked filenames:** `['Okwir_D_BBA_2024.pdf']`
- **Relevant (ground truth):** `['Okwir_D_BBA_2024.pdf']`
