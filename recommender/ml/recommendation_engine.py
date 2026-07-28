"""
TED Talks Hybrid Recommendation Engine
Uses REAL ted_main.csv dataset (2550 talks)
Combines:
  1. Content-Based Filtering  -> TF-IDF on title + description + tags
  2. Collaborative Filtering  -> Simulated user-rating matrix (SVD)
  3. Clustering               -> KMeans on TF-IDF vectors for topic grouping
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import os, warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, '..', '..', 'ted_main.csv')

# ─────────────────────────────────────────────
# Load & Clean Real Dataset
# ─────────────────────────────────────────────
def get_dataframe():
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=['name'], errors='ignore')
    df = df.rename(columns={'main_speaker': 'speaker'})
    df = df.reset_index(drop=True)

    # clean each column using list comprehensions to avoid pandas alignment bugs
    df['views']    = [int(v) if str(v) not in ('', 'nan') else 0 for v in df['views'].tolist()]
    df['tags']     = [str(t) if str(t) != 'nan' else '' for t in df['tags'].tolist()]
    df['description'] = [str(d) if str(d) != 'nan' else '' for d in df['description'].tolist()]
    df['duration'] = [int(d) if str(d) not in ('', 'nan') else 0 for d in df['duration'].tolist()]
    df['speaker_occupation'] = [str(o) if str(o) != 'nan' else 'Speaker' for o in df['speaker_occupation'].tolist()]
    df['event']    = [str(e) if str(e) != 'nan' else '' for e in df['event'].tolist()]
    df['url']      = [str(u) if str(u) != 'nan' else '' for u in df['url'].tolist()]

    # human-readable duration
    df['duration_fmt'] = [str(d // 60) + ' min' for d in df['duration'].tolist()]

    # combined content for TF-IDF
    df['content'] = [
        str(t) + ' ' + str(d) + ' ' + str(tg)
        for t, d, tg in zip(df['title'].values.tolist(), df['description'].values.tolist(), df['tags'].values.tolist())
    ]

    # top 500 by views for fast ML (increase to 1000+ on powerful machines)
    df = df.sort_values('views', ascending=False).head(500).reset_index(drop=True)
    df['id'] = range(1, len(df) + 1)

    return df[['id', 'title', 'speaker', 'speaker_occupation',
               'description', 'tags', 'views', 'duration',
               'duration_fmt', 'content', 'event', 'url']]


# ─────────────────────────────────────────────
# 1. Content-Based Filtering (TF-IDF)
# ─────────────────────────────────────────────
def build_tfidf_matrix(df):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        min_df=2
    )
    tfidf_matrix = vectorizer.fit_transform(df['content'])
    return vectorizer, tfidf_matrix


def content_based_recommendations(talk_id, df, tfidf_matrix, top_n=5):
    idx_list = df.index[df['id'] == talk_id].tolist()
    if not idx_list:
        return df.head(top_n).copy().assign(similarity_score=0.0)
    idx = idx_list[0]
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    cosine_sim[idx] = 0
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = [float(cosine_sim[i]) for i in top_indices]
    return results


# ─────────────────────────────────────────────
# 2. Collaborative Filtering (SVD)
# ─────────────────────────────────────────────
def build_user_rating_matrix(df, n_users=200):
    np.random.seed(42)
    n_talks = len(df)
    ratings = np.random.choice(
        [0, 0, 0, 1, 2, 3, 4, 5],
        size=(n_users, n_talks),
        p=[0.5, 0.1, 0.05, 0.1, 0.1, 0.08, 0.05, 0.02]
    ).astype(float)

    views_list = df['views'].tolist()
    max_views = max(views_list) if max(views_list) > 0 else 1
    for i in range(n_talks):
        popularity = min(views_list[i] / max_views, 1.0)
        mask = ratings[:, i] > 0
        ratings[mask, i] = np.clip(ratings[mask, i] + popularity * 2, 0, 5)

    return ratings


def build_svd_model(rating_matrix, n_components=30):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd.fit_transform(rating_matrix.T)
    return svd, normalize(latent_matrix)


def collaborative_recommendations(talk_id, df, latent_matrix, top_n=5):
    idx_list = df.index[df['id'] == talk_id].tolist()
    if not idx_list:
        return df.head(top_n).copy().assign(collab_score=0.0)
    idx = idx_list[0]
    scores = cosine_similarity([latent_matrix[idx]], latent_matrix).flatten()
    scores[idx] = 0
    top_indices = scores.argsort()[-top_n:][::-1]
    results = df.iloc[top_indices].copy()
    results['collab_score'] = [float(scores[i]) for i in top_indices]
    return results


# ─────────────────────────────────────────────
# 3. KMeans Clustering
# ─────────────────────────────────────────────
CLUSTER_LABELS = {
    0: "Science & Technology",
    1: "Psychology & Mind",
    2: "Society & Culture",
    3: "Health & Medicine",
    4: "Business & Economics",
    5: "Environment & Climate",
    6: "Education & Creativity",
    7: "Politics & Global Issues",
    8: "Art & Design",
    9: "Brain & Neuroscience",
}


def build_clusters(tfidf_matrix, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(tfidf_matrix)
    return kmeans, labels


# ─────────────────────────────────────────────
# 4. Hybrid Recommender
# ─────────────────────────────────────────────
class HybridRecommender:
    def __init__(self):
        print("[ML] Loading TED Talks dataset from ted_main.csv ...")
        self.df = get_dataframe()
        print(f"[ML] Loaded {len(self.df)} talks")

        print("[ML] Building TF-IDF matrix ...")
        self.vectorizer, self.tfidf_matrix = build_tfidf_matrix(self.df)

        print("[ML] Building user ratings matrix ...")
        self.rating_matrix = build_user_rating_matrix(self.df)

        print("[ML] Running SVD (collaborative filtering) ...")
        self.svd, self.latent_matrix = build_svd_model(self.rating_matrix)

        print("[ML] Running KMeans clustering ...")
        self.kmeans, cluster_labels = build_clusters(self.tfidf_matrix)

        self.df = self.df.copy()
        self.df['cluster'] = cluster_labels
        self.df['cluster_name'] = [CLUSTER_LABELS.get(int(c), 'General') for c in cluster_labels]
        print("[ML] Hybrid engine ready!")

    def recommend(self, talk_id, top_n=6, alpha=0.6):
        """
        Hybrid score = alpha * content_score + (1-alpha) * collab_score
        Uses self.df (which already has cluster_name) for both arms.
        """
        content_recs = content_based_recommendations(
            talk_id, self.df, self.tfidf_matrix, top_n=top_n * 2
        )
        collab_recs = collaborative_recommendations(
            talk_id, self.df, self.latent_matrix, top_n=top_n * 2
        )

        # Build score lookup dicts by talk id
        content_scores = dict(zip(
            content_recs['id'].tolist(),
            content_recs['similarity_score'].tolist()
        ))
        collab_scores = dict(zip(
            collab_recs['id'].tolist(),
            collab_recs['collab_score'].tolist()
        ))

        # Collect all candidate IDs (union of both)
        all_ids = set(content_scores.keys()) | set(collab_scores.keys())

        rows = []
        for cid in all_ids:
            cs = float(content_scores.get(cid, 0.0))
            col = float(collab_scores.get(cid, 0.0))
            hs = alpha * cs + (1 - alpha) * col
            rows.append({'id': cid, 'similarity_score': cs, 'collab_score': col, 'hybrid_score': hs})

        scores_df = pd.DataFrame(rows)

        # Merge scores with full talk metadata from self.df
        result = pd.merge(scores_df, self.df, on='id', how='left')
        result = result.sort_values('hybrid_score', ascending=False).head(top_n)
        return result

    def get_talk_by_id(self, talk_id):
        result = self.df[self.df['id'] == talk_id]
        return result.iloc[0] if not result.empty else None

    def search_talks(self, query, top_n=10):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        results = self.df.iloc[top_indices].copy()
        results['search_score'] = [float(scores[i]) for i in top_indices]
        return results[results['search_score'] > 0.01]

    def get_all_talks(self):
        return self.df

    def get_cluster_info(self):
        return self.df.groupby('cluster_name').size().reset_index(name='count').sort_values('count', ascending=False)


# Singleton instance
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = HybridRecommender()
    return _engine
