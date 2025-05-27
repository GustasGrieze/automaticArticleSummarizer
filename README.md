# Automatic Article Summarizer

## 1  Code overview (`main.py`)

* Loads a BART‑Large summarisation checkpoint (or another model you set in
  `MODEL_NAME`) and automatically moves it to GPU if one is detected.
* Restores punctuation for raw, lower‑cased texts using
  **deepmultilingualpunctuation** so the model receives well‑formed sentences.
* Generates **≥ 8‑sentence abstractive summaries** with beam search and dynamic
  length‑extension if the first attempt is too short.
* Lets the user either **pick an article from `tekstynas.txt`** *or* paste a
  **URL** – the script downloads the page, cleans the HTML and summarises the result.
* After every run the summary is appended to **`santrauka.txt`** for later
  inspection.

# 2   Quick‑start

```bash
pip install -r requirements.txt

python main.py
```

## 3  Corpus name & version

**Name:** *Social media Paragraphs*  
**Version:** v1.0

## 4  Goal & purpose

Provide a small, open corpus (≈ 12 k tokens) for experimenting with extractive / abstractive
summaries in the accompanying CLI tool.

## 5  Size & basic stats

| Metric            | Value           |
| ----------------- | --------------- |
| Documents         | 35 articles     |
| Tokens (clean)    | \~12 300        |
| Avg. tokens / doc | 352             |
| Languages         | English (100 %) |

## 6  Sources & distribution

| Source                                  | Share |
| --------------------------------------- | ----- |
| BBC News                                | 40 %  |
| Wikipedia                               | 35 %  |
| Manually‑curated web articles (various) | 25 %  |

## 7  Collection & cleaning pipeline

1. Tokenise sentences & words.
2. Lemmatise.
3. Remove stop‑words.
4. Strip control chars, normalise whitespace, NFC.
5. Deduplicate identical sentences/paragraphs.

## 8 Technical specification

* Encoding UTF‑8, Unix line‑endings.
* Average article fits within 1 024 BART tokens; safe for default 1 GiB RAM.
* Tested on Python 3.9
