# Resume ↔ Job Description Matching (POC)

This repository is a proof-of-concept (POC) matching engine that **scores and ranks resumes** against a single job description (JD). It's built for a Talent Acquisition workflow where the goal is to **quickly shortlist** the most relevant candidates with **clear "why" explanations**.

---

## 1) What This Does

### Input
- **Job description**: `data/job_description.txt`
- **Resumes folder**: `data/resumes/` (`.txt` or `.pdf` files)

### Output
- **Ranked results CSV**: `outputs/scores.csv`
- **Audit/debug text**:  
  - `outputs/extracted/` (PDF/TXT extracted text)  
  - `outputs/redacted/` (PII-redacted text, if enabled)

Each resume gets a **final score in [0.0, 1.0]** (higher = more relevant) plus explanation fields.

---

## 2) Technical Approach (Step-by-Step)

This POC uses a **hybrid ranker** + **practical gates** + optional **cross-encoder reranking**.

### Step A — Text Extraction + Preprocessing

1. **Load JD text** from file
2. **Load each resume**:
   - `.txt`: read as-is
   - `.pdf`: extract text using `src/extract.py`
3. **Normalize text** (lowercase, whitespace cleanup) via `src/preprocess.py`
4. **Optional**: `--redact_pii` removes common PII patterns (email/phone/URL) via `src/redact.py`
5. **Save** extracted + redacted text for audit/debug

**Why?**
- Real resumes arrive as PDFs
- Preprocessing reduces noise and makes matching more stable
- Redaction is a basic privacy safeguard for demos

---

### Step B — Lexical Relevance (BM25)

We compute **BM25 scores** using token overlap between JD tokens and each resume's tokens.

**Why?**
- Rewards exact must-have terms (tools, technologies, frameworks)
- Provides an interpretable keyword-based signal
- Catches hard requirements that semantic models might miss

---

### Step C — Semantic Relevance (Embeddings) with Chunking

1. **Embed the JD once** using a Sentence-Transformers bi-encoder
2. **Split each resume** into overlapping token chunks:
   - Default: **320 tokens** per chunk
   - Overlap: **80 tokens** between chunks
3. **Embed each chunk** and compute similarity to JD
4. **Aggregate** using:
   - Max similarity (default), or
   - Top-k average (configurable)

**Why Chunking?**
- Whole-document embeddings can dilute signal for long/noisy resumes
- Chunking ensures the most relevant section is captured
- Handles resumes with multiple roles or projects more effectively

---

### Step D — Hybrid Score

We **min-max scale** both BM25 and semantic scores to [0,1] within the batch and combine them:

```
hybrid_score = w_sem × semantic_score + w_bm25 × bm25_score
```

**Default weights:**
- `w_sem = 0.7` (semantic understanding)
- `w_bm25 = 0.3` (keyword matching)

**Important Note:**
This is a **ranking signal** within the given resume batch, not a calibrated probability.

---

### Step E — Practical Gates (Role Realism)

To reduce false positives (generic resumes scoring high), we apply:

#### 1) Must-Have Bucket Coverage (`src/skills.py`)
- Define role-specific keyword buckets (e.g., "languages", "frameworks", "cloud")
- If a resume misses **2+ buckets**, apply a multiplicative penalty
- Ensures candidates meet minimum technical requirements

#### 2) Seniority / Experience Penalty (`src/seniority.py`)
- Extract required years from JD (e.g., "3+ years", "5-7 years experience")
- Estimate resume years from:
  - Explicit mentions: "X years of experience"
  - Date ranges: employment history duration
- Apply penalty when resume appears under-qualified

**Why?**
- Recruiters care about must-haves and minimum experience
- These gates make ranking more practical and reduce irrelevant matches
- Prevents junior candidates from ranking above qualified seniors

---

### Step F — Explainability (Why This Resume Scored High)

We output two explanation fields for each resume:

1. **`top_terms`**: Top overlapping JD terms found in the resume (lexical explanation)
2. **`top_sentences`**: Top 3 resume sentences most semantically similar to the JD (semantic explanation)

**Why?**
- Recruiters always ask "why is this ranked high?"
- These fields answer that directly and build trust in the system
- Enables quick scanning of candidate relevance

---

### Optional Step G — Cross-Encoder Reranking (Top-N Only)

A **cross-encoder** reads JD + resume together and produces a more precise relevance score.

**Enabled with:**
```bash
--use_cross_encoder --rerank_top_n N
```

**Why Optional?**
- Cross-encoders are slower (attention over both texts)
- We rerank only the shortlist (top-N) for efficiency

---

## 3) Project Structure

```
ema-resume-matching-poc/
│
├── src/
│   ├── cli.py           # Main entrypoint: load → extract → redact → score → save CSV
│   ├── extract.py       # PDF/TXT extraction
│   ├── redact.py        # Basic PII redaction (email/phone/URL)
│   ├── preprocess.py    # Normalization + tokenization + chunking helpers
│   ├── matching.py      # BM25 + chunked embeddings + hybrid + rerank + explanations
│   ├── skills.py        # Must-have buckets + penalty logic
│   ├── seniority.py     # Years extraction + seniority penalty
│   └── evaluate.py      # Synthetic evaluation metrics (nDCG, Spearman, etc.)
│
├── data/
│   ├── job_description.txt    # Input: job description
│   ├── resumes/               # Input: candidate resumes (.txt or .pdf)
│   └── labels.csv             # Optional: ground truth labels for evaluation
│
├── outputs/
│   ├── scores.csv             # Output: ranked results with scores + explanations
│   ├── extracted/             # Extracted text from PDFs/TXTs
│   └── redacted/              # PII-redacted text (if --redact_pii enabled)
│
├── notebooks/
│   └── demo.ipynb             # Client-friendly end-to-end walkthrough
│
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 4) Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```


---

## 5) Run: Score & Rank Resumes

### Baseline Run 

```bash
python -m src.cli \
  --jd data/job_description.txt \
  --resumes data/resumes \
  --redact_pii \
  --output outputs/scores.csv
```

### Cross-Encoder Rerank

```bash
python -m src.cli \
  --jd data/job_description.txt \
  --resumes data/resumes \
  --output outputs/scores.csv \
  --use_cross_encoder \
  --rerank_top_n 5
```

### Advanced: Tuning Chunking Parameters

```bash
python -m src.cli \
  --jd data/job_description.txt \
  --resumes data/resumes \
  --output outputs/scores.csv \
  --chunk_size_tokens 320 \
  --chunk_overlap_tokens 80 \
  --semantic_top_k_avg 1     # 1 = max chunk; >1 = top-k average
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--jd` | Required | Path to job description file |
| `--resumes` | Required | Path to resumes folder |
| `--output` | Required | Path to output CSV |
| `--redact_pii` | False | Enable PII redaction |
| `--use_cross_encoder` | False | Enable cross-encoder reranking |
| `--rerank_top_n` | 10 | Number of top candidates to rerank |
| `--chunk_size_tokens` | 320 | Token size per chunk |
| `--chunk_overlap_tokens` | 80 | Overlap between chunks |
| `--semantic_top_k_avg` | 1 | Aggregation method for chunk scores |

### Outputs

After running, you'll find:

- **CSV**: `outputs/scores.csv` with columns:
  - `resume_id`: filename
  - `final`: hybrid score [0.0, 1.0]
  - `bm25`: lexical match score
  - `semantic`: embedding similarity score
  - `top_terms`: key matching terms
  - `top_sentences`: most relevant resume excerpts
  
- **Extracted text**: `outputs/extracted/` (for debugging PDF extraction)
- **Redacted text**: `outputs/redacted/` (if `--redact_pii` enabled)

---

## 6) Evaluate on Synthetic Labels

### Create Ground Truth Labels

Create `data/labels.csv` with:

**Label meanings:**
- `1.0` = Good match (strong candidate)
- `0.5` = Partial match (some relevant skills)
- `0.0` = Poor match (not qualified)

### Run Evaluation

```bash
python -m src.evaluate \
  --jd data/job_description.txt \
  --resumes data/resumes \
  --labels data/labels.csv \
  --redact_pii \
  --use_cross_encoder \
  --rerank_top_n 10
```

### Evaluation Metrics

This prints:

1. **Per-resume scores** with labels
2. **Spearman correlation**: How well scores correlate with labels
3. **nDCG@K**: Normalized Discounted Cumulative Gain (graded ranking quality)
   - nDCG@3: Quality of top 3 results
   - nDCG@5: Quality of top 5 results
   - nDCG@10: Quality of top 10 results

**Interpreting Results:**
- Spearman > 0.7: Strong correlation between scores and labels
- nDCG@K > 0.8: Excellent ranking quality
- nDCG@K 0.6-0.8: Good ranking quality
- nDCG@K < 0.6: Needs improvement

---

## 7) Notebook Demo (Client Presentation)

The notebook runs the full flow with visualizations:

1. Load JD and resumes
2. PDF extraction (with optional redaction)
3. Baseline ranking
4. Explanations (top_terms, top_sentences)
5. Cross-encoder reranking comparison
6. Evaluation metrics with charts

### Run Notebook

```bash
# Install notebook support
pip install notebook

# Start Jupyter
jupyter notebook

# Open: notebooks/demo.ipynb
```


---

## 8) If We Had a Larger Labeled Dataset

In hiring search, success is measured by **"good candidates appearing in the top-K"**.

### Recommended Metrics

1. **nDCG@K** (Normalized Discounted Cumulative Gain)
   - Best for graded relevance (0/0.5/1)
   - Accounts for position in ranking
   - Industry standard for search quality

2. **Precision@K**
   - Fraction of top-K that are good matches
   - Easy to interpret for recruiters
   - Example: "70% of top-10 are qualified"

3. **Recall@K**
   - How many good candidates we found in top-K
   - Ensures we don't miss qualified candidates
   - Critical for talent pipeline

4. **Optional Advanced Metrics**:
   - **MAP** (Mean Average Precision): For multiple queries
   - **MRR** (Mean Reciprocal Rank): When only first good match matters
   - **Confusion Matrix**: For binary classification analysis

### A/B Testing Framework

With production data, we'd measure:
- **Click-through rate**: Do recruiters open high-ranked resumes?
- **Interview conversion**: Do top-K candidates get interviewed?
- **Hire rate**: Do top-K candidates get hired?
- **Time-to-hire**: Does better ranking speed up hiring?

---

## 9) The Truth About This POC (Limitations)

### Current Limitations

1. **PDF Extraction**
   - Imperfect for complex layouts
   - Scanned/image PDFs may extract empty text
   - Tables and multi-column layouts can scramble text

2. **Scoring Interpretation**
   - Scores are **batch-relative**, not calibrated probabilities
   - A score of 0.8 means "80% of max in this batch", not "80% likely to be hired"
   - Different resume batches will have different score distributions

3. **Learning & Adaptation**
   - This is **unsupervised matching**
   - Production should learn from recruiter outcomes (clicks, interviews, hires)
   - No feedback loop to improve over time

4. **Bias & Fairness**
   - Basic PII redaction is not sufficient for production
   - Models may inherit biases from training data
   - Needs stronger fairness audits and debiasing techniques

5. **Scalability**
   - Cross-encoder is slow for large candidate pools
   - Current implementation is single-threaded
   - No caching or incremental indexing

6. **Domain Specificity**
   - Must-have buckets are manually defined
   - Seniority extraction uses simple heuristics
   - May not generalize across all job types

---

## 10) Next Improvements

### Short-Term (1-3 months)

1. **Better PDF Handling**
   - Layout-aware parsing (preserve structure)
   - OCR fallback for scanned resumes
   - Table extraction for skills matrices

2. **Enhanced Entity Extraction**
   - Job titles with seniority (Senior, Lead, Principal)
   - Technologies and tools (with versions)
   - Certifications and degrees
   - Years of experience per technology

3. **Stronger Must-Have Gating**
   - Role-specific mandatory skill buckets
   - Weighted requirements (nice-to-have vs. must-have)
   - Industry-specific terminology dictionaries

### Medium-Term (3-6 months)

4. **Learning-to-Rank (LTR)**
   - Collect real outcomes: screened, interviewed, hired
   - Train ranking model on actual recruiter decisions
   - A/B test against baseline

5. **Bias Mitigation**
   - Fairness metrics across demographics
   - Adversarial debiasing techniques
   - Blind screening options

6. **Multi-JD Support**
   - Batch process multiple job descriptions
   - Cross-JD candidate recommendations
   - Pipeline management features

### Long-Term (6-12 months)

7. **Production Architecture**
   - **Stage 1**: BM25 or Elasticsearch for fast candidate retrieval
   - **Stage 2**: ANN (Approximate Nearest Neighbors) with embeddings
   - **Stage 3**: Cross-encoder rerank top-100 → top-10
   - **Stage 4**: Human review with explanations

8. **Advanced Features**
   - Skills gap analysis: "Candidate has X, needs Y"
   - Career trajectory prediction
   - Interview question generation based on gaps
   - Automated resume parsing with entity linking

9. **MLOps & Monitoring**
   - Model versioning and A/B testing
   - Ranking quality dashboards
   - Data drift detection
   - Automated retraining pipelines

---

**Built with ❤️ for smarter hiring**
