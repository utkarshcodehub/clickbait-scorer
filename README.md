# Indian News Clickbait Scorer

**Day 03 of the 21-Day Build Challenge**

🔗 **Live:** [clickbait-scorer.streamlit.app](https://clickbait-scorer.streamlit.app)

---

## What It Does

Paste any Indian news headline and get a clickbait score with the reasoning behind it. Built on a classifier trained on real headlines from Indian outlets — high-clickbait sources (Times of India) vs low-clickbait (The Hindu, The Wire).

Not a vibe check. An actual ML model with feature-level transparency.

---

## Features

- **Single headline scorer** — paste any headline, get a score and a feature DNA breakdown
- **Bulk URL scraper** — paste multiple article URLs, the app scrapes and scores all of them
- **Clickbait signal highlighting** — shows which specific words/patterns drove the score
- **Share card** — generates a downloadable PNG of your headline's result

---

## Stack

| Tool | Use |
|------|-----|
| Python | Core logic |
| scikit-learn | TF-IDF + logistic regression classifier |
| BeautifulSoup | Headline scraping from URLs |
| Streamlit | UI and deployment |
| Plotly | Score visualisation |
| joblib | Model serialisation |

---

## How to Run Locally

```bash
git clone https://github.com/yourusername/21-day-build-challenge
cd day-03-clickbait-scorer
pip install -r requirements.txt
streamlit run app.py
```

---

## How the Model Works

- **Training data:** Headlines scraped from TOI (labelled clickbait) vs The Hindu/The Wire (labelled non-clickbait)
- **Features:** TF-IDF on unigrams and bigrams, headline length, punctuation density, capital letter ratio, question/exclamation presence
- **Model:** Logistic regression — interpretable by design, each feature's coefficient is surfaced in the DNA breakdown

---

## What I Learned

- Indian news has very distinct clickbait patterns compared to Western datasets — emotional superlatives, "shocking", "you won't believe" patterns are common but the model also picks up on vague pronoun usage ("this politician", "a celeb")
- Scraping headline text from article URLs requires handling paywalls and varying HTML structures — a fallback to `<title>` tag works for most cases
- Making model internals visible (the DNA breakdown) makes the tool feel like a learning instrument rather than a black box
