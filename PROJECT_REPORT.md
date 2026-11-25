# Movie Recommendation System Using Content-Based, Collaborative and Hybrid Filtering

## Abstract

This minor project develops an interactive movie recommendation system using Python, Pandas, NumPy, SciPy, scikit-learn, Streamlit and optional Surprise. The supplied dataset contains 10,329 movie records with titles, identifiers and genre labels. The system converts complete genre tokens into TF-IDF vectors and ranks candidates by cosine similarity to selected favorites. A sparse item-based collaborative model and optional SVD model support explicit user ratings when supplied. A weighted hybrid approach combines both signals. The system includes exploratory analysis and reproducible evaluation with clearly stated limitations.

## 1. Define recommendation type

Content-based filtering recommends items with similar attributes. Collaborative filtering learns from shared user ratings. The hybrid method combines both when both are available. Content-based filtering is the default because the supplied data contains no user interactions.

## 2. Collect data

The input catalog has 10,329 rows and three columns: movieId (integer identifier), title (movie title), genres (pipe-delimited categories). The original file is bundled unchanged. There are no nulls in the input, no duplicate movie IDs, two duplicate-title entries beyond their first occurrences, seven movies marked `(no genres listed)`, and three titles without a parseable trailing release year. A compatible real ratings file is required to measure user preference patterns; it is not supplied. IDs, not titles, are used to distinguish movies.

## 3. Preprocess data

Pandas validates required columns, strips titles, removes invalid/duplicate IDs, fills missing genre values and extracts nullable years. Genre strings are split on the pipe character, preserving terms such as Sci-Fi and Film-Noir as single tokens. TF-IDF vectors are L2-normalized. Ratings, if uploaded, are restricted to the catalog and the 0.5–5 scale; duplicate user/movie pairs retain the latest timestamp or file row. The hybrid score scales predicted ratings to 0–1 using (rating − 0.5)/4.5.

## 4. Exploratory data analysis

Genre counts (multi-label, so totals exceed the number of movies): Drama 5,220; Comedy 3,515; Thriller 2,187; Romance 1,788; Action 1,737; Crime 1,440; Adventure 1,164; Horror 1,001; Sci-Fi 860. There are 19 actual genre categories. The interface charts all genres and the release-year distribution. When real ratings are supplied, it also shows rating frequency, ratings per user, user count, rated movie count, average rating, observed user-item density and catalog coverage. User-item analysis is unavailable from metadata alone.

## 5. Model user behavior

For content filtering, a user selects one or more favorites. Their profile is the normalized mean of the selected TF-IDF vectors. Cosine similarity is the dot product of unit-length vectors. For collaborative filtering, users' explicit ratings form a sparse user-item matrix R. Similarity between items is the cosine between their columns. Predicted ratings are weighted means of the user's known ratings using these item similarities. This is a simple non-mean-centered baseline; raw positive ratings can amplify popularity and rating-scale biases. Optional Surprise SVD learns user and item latent factors and biases; it uses a fixed seed for reproducible initialization, not to generate fake observations.

## 6. Build recommendation engine

Architecture:

```text
movies.csv -> clean catalog -> sparse TF-IDF -> content similarity ---+
                                                                     +--> rank/filter --> Streamlit
ratings.csv (optional) -> clean ratings -> item cosine or SVD --------+
```

Hybrid score = alpha × content cosine + (1 − alpha) × normalized rating prediction. Default alpha is 0.6, a design choice, not an empirically optimized value. Only collaboratively supported unseen candidates enter hybrid ranking. Favorite and already-rated items are removed from recommendations. Equal scores are ordered by ascending movie ID for reproducibility.

## 7. Generate recommendations

With Toy Story (1995) as a favorite, the actual top five returned by the supplied dataset are Antz (1998), Toy Story 2 (1999), Adventures of Rocky and Bullwinkle, The (2000), Emperor's New Groove, The (2000), and Monsters, Inc. (2001). Each has the same five genre labels and a cosine score of approximately 1. This reflects identical metadata features, not guaranteed user satisfaction.

## 8. Evaluate recommendations

A deterministic metadata experiment uses 120 evenly spaced multi-genre movies, hides their alphabetically first genre, and checks whether the top ten other movies have the hidden genre. Precision@10 = relevant recommendations / 10; recall@10 = relevant recommendations / catalog items with the hidden genre; hit rate = fraction of queries with at least one relevant result. The baseline ranks by movie ID. Results on the bundled data:

| Method | Precision@10 | Recall@10 | Hit rate |
|---|---:|---:|---:|
| TF-IDF cosine | 0.0667 | 0.000295 | 0.1500 |
| Movie ID baseline | 0.2675 | 0.001528 | 0.9333 |

The content method underperforms the baseline on this proxy. This experiment assesses hidden genre retrieval, not real preference prediction. Alphabetical genre removal, catalog order, genre prevalence and coarse labels can bias results. The values must not be reported as user recommendation accuracy. Future evaluation should use actual interactions and compare ranking metrics on held-out preferences.

Optional item-cosine evaluation holds out one rating for up to 100 users with at least three valid ratings (latest by timestamp if present, otherwise last file row). It trains on the remaining interactions, computes MAE and RMSE against observed ratings, reports neighbor coverage, and uses the user's training mean when no neighbors exist. It also reports a user-mean baseline. This split is per-user, not a globally chronological simulation. SVD and hybrid ranking are not evaluated by the bundled evaluation script. Real ratings metrics are unavailable until ratings are provided.

## 9. Optimize scalability

Store sparse vectors and compute only query-to-catalog similarities; avoid an all-movies squared similarity matrix. Collaborative similarities are computed for the selected user's rated items only. Cache parsed datasets, model objects and evaluation results with Streamlit. Future improvements include approximate nearest neighbors for richer features, validation-based hybrid tuning, chronological ranking evaluation, stronger regularized CF, and model persistence for larger datasets.

## 10. Document design and limitations

The source modules separate UI, model, EDA and evaluation. Content recommendations are limited by genre-only attributes and cannot distinguish films with identical genre vectors. New users without a profile and new items without ratings create cold-start issues. Ratings are required for true collaborative behavior. Hybrid effectiveness is not established. The optional Surprise package has environment dependencies. No external API, poster service, or login is necessary.

## Conclusion

The implemented system provides working metadata-based movie discovery with a real-data extension for collaborative and hybrid methods. The metadata test provides a transparent baseline comparison and demonstrates why user-level data and richer features are needed before claiming improved recommendation accuracy.

## References

- scikit-learn documentation: TF-IDF and cosine similarity — https://scikit-learn.org/stable/
- Surprise documentation — https://surprise.readthedocs.io/
- Streamlit documentation — https://docs.streamlit.io/
- GroupLens MovieLens datasets — https://grouplens.org/datasets/movielens/ (possible compatible ratings source; verify matching release)
