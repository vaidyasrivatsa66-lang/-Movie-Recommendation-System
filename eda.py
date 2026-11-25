"""Catalog EDA and optional rating interaction summaries."""
from recommender import load_movies, genre_tokens

def genre_counts(m):
    return m.genres.map(genre_tokens).explode().dropna().value_counts().rename('movies')

def year_counts(m):
    return m.year.dropna().astype(int).value_counts().sort_index().rename('movies')

def interaction_summary(r, m):
    u=r.userId.nunique(); i=r.movieId.nunique()
    return {'users':u,'rated_movies':i,'ratings':len(r),'mean_rating':float(r.rating.mean()),'observed_matrix_density':len(r)/(u*i),'catalog_coverage':i/len(m)}

if __name__=='__main__':
    m=load_movies(); print(m.head()); print(genre_counts(m)); print(year_counts(m).tail())
