# 🎤 TED Talks Recommendation System
### A Mini Project · Django + Machine Learning

A hybrid recommendation engine built with Django, combining **NLP content filtering**, **collaborative filtering (SVD)**, and **KMeans clustering** to recommend TED Talks.

---

## 🧠 ML Architecture

```
User clicks a Talk
       ↓
┌──────────────────────────────────────────┐
│           HYBRID RECOMMENDER             │
│                                          │
│  ① Content-Based (weight: 60%)          │
│     TF-IDF on title + description + tags │
│     → Cosine Similarity                  │
│                                          │
│  ② Collaborative Filtering (weight: 40%)│
│     Simulated user ratings matrix        │
│     → Truncated SVD (20 components)      │
│     → Cosine Similarity on latent space  │
│                                          │
│  ③ KMeans Clustering (8 topics)         │
│     Groups talks into topic clusters     │
│     Used for browsing + filtering        │
│                                          │
│  Hybrid Score = 0.6×content + 0.4×collab│
└──────────────────────────────────────────┘
       ↓
  Top 6 Recommendations
```

---

## 📁 Project Structure

```
tedtalks_recommender/
│
├── manage.py
├── requirements.txt
│
├── tedtalks_project/          # Django project config
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
│
└── recommender/               # Main Django app
    ├── views.py               # All API endpoints
    ├── urls.py                # URL routing
    ├── ml/
    │   └── recommendation_engine.py   # 🧠 Core ML logic
    └── templates/
        └── recommender/
            └── index.html     # Frontend UI
```

---

## ⚙️ Setup & Run

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Run migrations
```bash
python manage.py migrate
```

### 3. Start the server
```bash
python manage.py runserver
```

### 4. Open in browser
```
http://127.0.0.1:8000/
```

---

## 🌐 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Main UI |
| `GET /recommend/<id>/` | Get hybrid recommendations for a talk |
| `GET /search/?q=<query>` | Search talks with TF-IDF |
| `GET /talk/<id>/` | Get single talk details |

---

## 🔬 Key ML Concepts Used

| Concept | Library | Purpose |
|---|---|---|
| TF-IDF Vectorizer | scikit-learn | Convert text to numerical features |
| Cosine Similarity | scikit-learn | Measure similarity between talks |
| TruncatedSVD | scikit-learn | Dimensionality reduction on ratings |
| KMeans | scikit-learn | Cluster talks into topic groups |
| Pandas DataFrame | pandas | Data manipulation |
| NumPy | numpy | Matrix operations |

---

## 📊 Dataset

- **50 curated TED Talks** with title, speaker, description, tags, views
- **8 topic clusters**: Psychology, Leadership, Technology, Health, Education, Communication, Science, Society
- Simulated user ratings matrix (100 users × 50 talks)

---

## 🚀 Extensions for Students

1. **Use real TED dataset** from Kaggle (ted_main.csv)
2. **Add user accounts** with Django auth for real collaborative filtering
3. **Deploy to Heroku/Railway** with Gunicorn + PostgreSQL
4. **Add more ML models**: Word2Vec, BERT embeddings
5. **A/B test** content vs collaborative vs hybrid approaches

---

## 👨‍🏫 Concepts to Explain in Class

- What is a **recommendation system**? (Amazon, Netflix, YouTube)
- **Content-based vs Collaborative** filtering trade-offs
- What is **TF-IDF** and why it works for text?
- What is **SVD** and how it finds latent patterns?
- What is **KMeans clustering** and elbow method?
- How to **combine signals** with weighted hybrid scoring?
