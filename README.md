# Trial-and-matcher-assitant
**TrialMatch Helper** is a *first-pass screening* tool for clinical trials. It asks a few simple profile questions (age, sex, city, condition, current meds, etc.), searches a curated set of **ClinicalTrials.gov** studies, and returns a short, explainable decision for each trial:

**`Likely` | `Unlikely` | `Need-Info`**, plus a brief “why/why-not” and the official **NCT link** to verify details.

> **Not medical advice.** It’s a fast pre-screen to narrow options before a clinician reviews the official criteria.

---

## Problem it solves
- Patients often apply to trials they don’t qualify for (age ranges, excluded medications, required lab values).
- Clinicians need a quick, auditable triage rather than reading every full trial page.

**Goal:** Save time and improve transparency by giving a grounded, explainable first pass.

---

## How it works (workflow)

1. **Light intake**  
   The user provides a minimal profile like:
   ```json
   { "age": 45, "sex": "M", "city": "Istanbul", "condition": "Type 2 Diabetes", "meds": ["metformin"] }
   ```

2. **Search & retrieval**  
   The system queries a local vector index built from **ClinicalTrials.gov** snapshots to pull the most relevant trials for the stated condition and location context.

3. **Eligibility parsing**  
   It focuses on the trials’ inclusion/exclusion criteria and key structured fields (e.g., recruiting status, locations).

4. **Rule-based matching**  
   It checks explicit, interpretable conditions (e.g., minimum/maximum age, sex requirements, common medication exclusions, required labs if present in text).

5. **Explainable decisions**  
   For each candidate trial, it returns:
   - a decision: **Likely / Unlikely / Need-Info**  
   - short reasons (e.g., *“age ok ✓”*, *“drug allowed ✓”*, *“need HbA1c?”*)  
   - the official **NCT link** for verification

6. **Guardrails**  
   Output is normalized to the fixed decision set, includes a medical disclaimer, and avoids unverifiable claims.

---

## What the output looks like

A ranked list of trials, each with decision + rationale + links, plus a concise summary:

```json
{
  "trials": [
    {
      "nct": "NCT0123456",
      "score": 0.82,
      "decision": "Likely",
      "why": ["age ok ✓", "drug allowed ✓", "need HbA1c ?"],
      "links": ["https://clinicaltrials.gov/study/NCT0123456"]
    }
  ],
  "summary": "Top trials ranked by fit and relevance.",
  "disclaimer": "Not medical advice."
}
```

---

## What it **is** vs. what it **is not**

- **Is:** a transparent, auditable **pre-screen** that narrows choices and explains why.  
- **Is not:** a diagnostic tool or a substitute for physician judgment. Final eligibility always depends on full criteria review and clinical evaluation.

---

## Data and assumptions

- **Source:** snapshots of **ClinicalTrials.gov** studies.  
- **Indexing:** trials are embedded and stored in a local vector index for fast retrieval.  
- **Coverage & freshness:** as the underlying snapshots evolve, the index should be refreshed to reflect the latest criteria and recruiting status.

---

## Why this design adds value

- **Time savings:** quickly filters out obviously ineligible trials.  
- **Transparency:** every decision comes with “why/why-not” and a link to the official NCT record.  
- **Practicality:** simple inputs, explainable rules, and verifiable outputs.  
- **Extendable:** supports adding more conditions, criteria extractors, or location/radius filters over time.

---

## Limitations to keep in mind

- If key fields are missing (e.g., specific lab values), the tool may return **`Need-Info`**.  
- Criteria can be nuanced; always confirm on the official NCT page and with the study team.  
- The tool prioritizes **explicit** criteria matching; ambiguous or complex medical edge cases require clinical review.


## Python Requirements

- `fastapi`
- `uvicorn`
- `dotenv`
- `langchain-openai`
- `langgraph`
- `langchain-community`
- `langfuse`
- `streamlit`
- `transformers`
- `torch`
- `bitsandbytes`
- `accelerate`
- `faiss-cpu`

## Dockerfile Facts

- **Base image:** `python:3.11-slim AS base`
- **WORKDIR:** `/app`
- **EXPOSE:** 8000
- **CMD:** ["uvicorn", "trial_matcher_fastapi:app", "--reload","--host", "127.0.0.1", "--port", "8000"]

## Environment Variables 
- `AIRFLOW_HOME`
- `BRIGHT_API_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SERCET_KEY`
- `OPENAI_API_KEY`
- `QD_API_KEY`
- `QD_END_POINT`


## FastAPI Routes

- `GET /match`
- `POST /match`

## Pydantic Models 
**Match**
- query: str
- lang: str

## Model/Tokenizer References 

- `/app/Qwen3-4B`
- `Qwen3-4B`

## Vector Index Artifacts

- `trial_vdb.faiss`: present
- `trial_vdb.pkl`: present
