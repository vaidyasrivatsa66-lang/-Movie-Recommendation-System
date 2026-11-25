# Movie Discovery — Movie Recommendation System

A Data Science minor project using the supplied catalog of 10,329 movies. The Streamlit interface supports content-based filtering immediately and genuine collaborative/hybrid filtering when real ratings are uploaded.

## Run on Windows

Use Python 3.11 (recommended for the optional Surprise package). Open PowerShell or Command Prompt inside the extracted project folder:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed in the terminal (usually http://localhost:8501). No API key is needed. The optional `run_windows.bat` uses the Python interpreter on your PATH; the commands above use the virtual environment explicitly.

To enable Surprise SVD instead of the built-in item-cosine algorithm:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-surprise.txt
```

If installation of scikit-surprise requires a compiler on your system, use the built-in item-cosine option instead. Do not install Surprise under an incompatible NumPy version; requirements pin NumPy below 2.

## Use the app

1. In Recommend, select one or more favorites and click Generate recommendations.
2. Explore dataset shows genre and release-year plots.
3. Evaluate runs the metadata experiment and, with ratings, the rating prediction test.
4. Optional: upload a real `ratings.csv` whose movie IDs match this catalog, then select Collaborative or Hybrid and an existing user ID. Expected headers: `userId,movieId,rating` with optional `timestamp`. Ratings must be on the 0.5–5 scale. Prefer the ratings file from the same MovieLens release as the supplied catalog; the exact release is not independently established here.
5. Hybrid mode accepts a content weight and blends the two scores. Movies without collaborative support are excluded. Export the recommendation table if needed.

No generated ratings are included. The tiny explicitly hand-written fixture in the unit tests is for software correctness only, not real data or model performance evidence.

## Files

- `app.py`: Streamlit GUI
- `recommender.py`: preprocessing, content engine, item cosine, optional Surprise SVD, hybrid ranking
- `eda.py`: exploratory analysis
- `evaluation.py`: reproducible metadata and optional held-out rating evaluation
- `data/movies.csv`: supplied dataset, unchanged
- `PROJECT_REPORT.md`: report covering all ten required steps
- `tests/test_engine.py`: unit tests
- `requirements.txt`, `requirements-surprise.txt`: dependencies

Run tests: `python -m unittest discover -s tests -v` (use the virtual environment interpreter if applicable).

Run standalone EDA and metadata evaluation: `python eda.py` and `python evaluation.py`.

## Verification status

Core engine, EDA helpers, and evaluation were executed in the creation environment; five unit tests passed, including collaborative/hybrid logic using a small test fixture. Python source compilation passed. Streamlit and Surprise are not installed in that environment, so browser rendering and SVD training have not been runtime-tested. No real ratings were provided, so no real MAE/RMSE or hybrid accuracy is claimed.
