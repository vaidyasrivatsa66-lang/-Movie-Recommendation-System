"""Sparse, ID-safe content, collaborative and hybrid recommendation engines."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

NO_GENRE = '(no genres listed)'
DATA = Path(__file__).parent / 'data' / 'movies.csv'

def genre_tokens(s):
    return [g for g in str(s).split('|') if g and g != NO_GENRE]

def load_movies(source=DATA):
    m = pd.read_csv(source)
    if not {'movieId','title','genres'}.issubset(m.columns):
        raise ValueError('Movies need movieId, title and genres.')
    m = m[['movieId','title','genres']].copy()
    m['movieId'] = pd.to_numeric(m.movieId, errors='coerce')
    m = m.dropna(subset=['movieId','title'])
    m = m[m.movieId.mod(1).eq(0)].copy()
    m['movieId'] = m.movieId.astype('int64')
    m['title'] = m.title.astype(str).str.strip()
    m = m[m.title.ne('')].drop_duplicates('movieId').sort_values('movieId').reset_index(drop=True)
    m['genres'] = m.genres.fillna(NO_GENRE).astype(str)
    m['year'] = pd.to_numeric(m.title.str.extract(r'\((\d{4})\)\s*$')[0], errors='coerce').astype('Int64')
    return m

def load_ratings(source, movies):
    r = pd.read_csv(source)
    if not {'userId','movieId','rating'}.issubset(r.columns):
        raise ValueError('Ratings need userId, movieId, rating (optional timestamp).')
    r = r.copy()
    for c in ['userId','movieId','rating']:
        r[c] = pd.to_numeric(r[c], errors='coerce')
    r = r.dropna(subset=['userId','movieId','rating'])
    r = r[r.userId.mod(1).eq(0) & r.movieId.mod(1).eq(0) & r.rating.between(0.5,5) & r.movieId.isin(movies.movieId)].copy()
    r[['userId','movieId']] = r[['userId','movieId']].astype('int64')
    if 'timestamp' in r:
        r['timestamp'] = pd.to_numeric(r.timestamp, errors='coerce')
        r = r.sort_values('timestamp', kind='stable', na_position='first')
    r = r.drop_duplicates(['userId','movieId'], keep='last').reset_index(drop=True)
    if r.empty:
        raise ValueError('No valid ratings matching the movie catalog (rating scale 0.5–5).')
    return r

class ContentEngine:
    def __init__(self, movies):
        self.movies = movies.reset_index(drop=True)
        self.index = {int(mid):i for i,mid in enumerate(self.movies.movieId)}
        self.vectorizer = TfidfVectorizer(analyzer=genre_tokens, dtype=np.float32)
        self.matrix = self.vectorizer.fit_transform(self.movies.genres)
    def scores(self, favorites):
        rows = [self.index[int(mid)] for mid in dict.fromkeys(favorites) if int(mid) in self.index]
        if not rows:
            raise ValueError('Select at least one movie.')
        profile = normalize(csr_matrix(self.matrix[rows].mean(axis=0)))
        if profile.nnz == 0:
            raise ValueError('Selected movies have no usable genre metadata.')
        return np.clip((self.matrix @ profile.T).toarray().ravel(), 0, 1)
    def rank(self, scores, exclude=(), n=10):
        s = np.asarray(scores, dtype=float).copy()
        for mid in exclude:
            if int(mid) in self.index: s[self.index[int(mid)]] = -np.inf
        valid = np.flatnonzero(np.isfinite(s))
        order = valid[np.lexsort((self.movies.movieId.to_numpy()[valid], -s[valid]))][:n]
        result = self.movies.iloc[order][['movieId','title','genres']].copy()
        result['score'] = s[order]
        return result.reset_index(drop=True)
    def recommend(self, favorites, n=10):
        s = self.scores(favorites)
        s[s <= 0] = -np.inf
        return self.rank(s, favorites, n)

class CollaborativeEngine:
    """Item cosine over sparse explicit rating vectors; no all-pairs matrix."""
    def __init__(self, movies, ratings):
        self.movies = movies.reset_index(drop=True)
        self.ratings = ratings
        self.index = {int(mid):i for i,mid in enumerate(self.movies.movieId)}
        self.users = {int(uid):i for i,uid in enumerate(sorted(ratings.userId.unique()))}
        cols = ratings.movieId.map(self.index).to_numpy()
        rows = ratings.userId.map(self.users).to_numpy()
        self.ui = csr_matrix((ratings.rating.to_numpy(dtype=np.float32), (rows,cols)), shape=(len(self.users),len(movies)))
        self.items = normalize(self.ui.T.tocsr())
        self.mean = float(ratings.rating.mean())
    def seen(self, uid):
        if uid not in self.users: return []
        return self.movies.movieId.iloc[self.ui.getrow(self.users[uid]).indices].tolist()
    def scores(self, uid):
        if uid not in self.users: raise ValueError('Unknown user ID.')
        row = self.ui.getrow(self.users[uid])
        # Unseen items have zero for the active user; self-rated items are excluded below.
        sim = self.items @ self.items[row.indices].T
        denom = np.asarray(sim.sum(axis=1)).ravel()
        numer = np.asarray(sim @ row.data).ravel()
        result = np.full(len(self.movies), np.nan)
        np.divide(numer, denom, out=result, where=denom>1e-8)
        result[row.indices] = np.nan
        return result
    def fallback(self, uid):
        return float(self.ui.getrow(self.users[uid]).data.mean()) if uid in self.users else self.mean

class SurpriseEngine:
    """Optional matrix-factorization backend for explicit ratings."""
    def __init__(self, movies, ratings):
        try:
            from surprise import Dataset, Reader, SVD
        except ImportError as e:
            raise ImportError('Install requirements-surprise.txt to enable SVD.') from e
        self.movies = movies.reset_index(drop=True)
        self.ratings = ratings
        self.users = set(ratings.userId.unique())
        self.known = set(ratings.movieId.unique())
        train = Dataset.load_from_df(ratings[['userId','movieId','rating']], Reader(rating_scale=(0.5,5))).build_full_trainset()
        self.model = SVD(n_factors=30, n_epochs=20, random_state=42)
        self.model.fit(train)
    def seen(self, uid):
        return self.ratings.loc[self.ratings.userId.eq(uid),'movieId'].tolist()
    def scores(self, uid):
        if uid not in self.users: raise ValueError('Unknown user ID.')
        seen = set(self.seen(uid))
        return np.array([self.model.predict(uid, mid).est if mid in self.known and mid not in seen else np.nan for mid in self.movies.movieId])

def recommend_collaborative(content, collaborative, uid, n=10):
    return content.rank(collaborative.scores(uid), collaborative.seen(uid), n)

def recommend_hybrid(content, collaborative, uid, favorites, alpha=0.6, n=10):
    if not 0 <= alpha <= 1: raise ValueError('Weight must be between 0 and 1.')
    c = content.scores(favorites)
    r = collaborative.scores(uid)
    # Only rank movies supported by the collaborative engine; no invented ratings.
    s = alpha*c + (1-alpha)*np.clip((r-0.5)/4.5,0,1)
    return content.rank(s, set(favorites)|set(collaborative.seen(uid)), n)
