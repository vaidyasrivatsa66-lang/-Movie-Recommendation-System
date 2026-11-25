"""Deterministic metadata proxy and optional held-out ratings evaluation."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from recommender import ContentEngine, CollaborativeEngine, genre_tokens, load_movies

def content_evaluation(engine, k=10, limit=120):
    m = engine.movies
    tags = [set(genre_tokens(s)) for s in m.genres]
    eligible = [i for i,t in enumerate(tags) if len(t)>=2]
    chosen = np.linspace(0,len(eligible)-1,min(limit,len(eligible)),dtype=int) if eligible else []
    records=[]
    for pos in chosen:
        i = eligible[pos]
        # Alphabetical choice makes the experiment reproducible, but may bias genre frequencies.
        held = sorted(tags[i])[0]
        query = '|'.join(sorted(tags[i]-{held}))
        v = engine.vectorizer.transform([query])
        scores = (engine.matrix @ v.T).toarray().ravel()
        scores[i] = -np.inf
        valid = np.flatnonzero(np.isfinite(scores))
        ranked = valid[np.lexsort((m.movieId.to_numpy()[valid],-scores[valid]))][:k]
        baseline = valid[:k]
        relevant = np.array([held in t for t in tags]); relevant[i]=False
        for method, picks in [('TF-IDF cosine',ranked),('Movie ID baseline',baseline)]:
            hits = int(relevant[picks].sum())
            records.append((method,hits/len(picks),hits/max(1,relevant.sum()),int(hits>0)))
    if not records: return pd.DataFrame(columns=['method','precision_at_k','recall_at_k','hit_rate','queries'])
    d=pd.DataFrame(records,columns=['method','precision_at_k','recall_at_k','hit_rate'])
    out=d.groupby('method',sort=False).mean().reset_index()
    out['queries']=len(chosen)
    return out

def ratings_evaluation(movies, ratings, limit=100):
    # load_ratings sorts by timestamp when provided; otherwise preserve file order.
    counts=ratings.groupby('userId').size()
    users=counts[counts>=3].index[:limit]
    test=ratings[ratings.userId.isin(users)].groupby('userId',sort=False).tail(1)
    if test.empty: return None
    train=ratings.drop(test.index)
    cf=CollaborativeEngine(movies,train)
    actual=[]; pred=[]; baseline=[]; supported=0
    for row in test.itertuples():
        s=cf.scores(row.userId)[cf.index[row.movieId]]
        b=cf.fallback(row.userId)
        supported+=int(np.isfinite(s))
        actual.append(row.rating); pred.append(float(s) if np.isfinite(s) else b); baseline.append(b)
    a=np.array(actual)
    def metrics(p):
        e=np.array(p)-a
        return {'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e**2)))}
    return {'test_ratings':len(a),'neighbor_coverage':supported/len(a),'item_cosine_with_fallback':metrics(pred),'user_mean_baseline':metrics(baseline)}

if __name__=='__main__':
    print(content_evaluation(ContentEngine(load_movies())).to_string(index=False))
