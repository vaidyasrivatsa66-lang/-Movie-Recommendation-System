import io
import streamlit as st
from recommender import (load_movies, load_ratings, ContentEngine, CollaborativeEngine,
                         SurpriseEngine, recommend_collaborative, recommend_hybrid)
from eda import genre_counts, year_counts, interaction_summary
from evaluation import content_evaluation, ratings_evaluation

st.set_page_config(page_title='Movie Discovery', page_icon='🎬', layout='wide')
@st.cache_data
def movies_data(): return load_movies()
@st.cache_resource
def content_model(): return ContentEngine(movies_data())
@st.cache_data
def ratings_data(raw): return load_ratings(io.BytesIO(raw), movies_data())
@st.cache_resource
def cf_model(raw, method):
    return (SurpriseEngine if method=='Surprise SVD' else CollaborativeEngine)(movies_data(),ratings_data(raw))
@st.cache_data
def meta_eval(): return content_evaluation(content_model())
@st.cache_data
def rate_eval(raw): return ratings_evaluation(movies_data(),ratings_data(raw))

m=movies_data(); content=content_model()
labels={int(r.movieId):f'{r.title} · ID {r.movieId}' for r in m.itertuples()}
st.title('🎬 Movie Discovery')
st.caption('Personalized recommendations · Data Science minor project')
a,b,c=st.columns(3); a.metric('Movies',f'{len(m):,}'); b.metric('Genres',len(genre_counts(m))); c.metric('Missing release years',int(m.year.isna().sum()))
with st.sidebar:
    st.header('Settings')
    upload=st.file_uploader('Optional ratings.csv',type=['csv'],help='Columns: userId, movieId, rating; optional timestamp. Ratings 0.5–5.')
    raw=upload.getvalue() if upload else None
    ratings=None
    if raw:
        try:
            ratings=ratings_data(raw); st.success(f'{len(ratings):,} valid ratings loaded')
        except Exception as e: st.error(f'Invalid ratings file: {e}')
    else: st.info('Content mode works with the bundled dataset. Upload real ratings to unlock collaborative and hybrid modes.')
    n=st.slider('Number of recommendations',5,30,10)
    page=st.radio('Navigate',['Recommend','Explore dataset','Evaluate','About'])

if page=='Recommend':
    st.subheader('Find your next movie')
    methods=['Content-based']+(['Collaborative','Hybrid'] if ratings is not None else [])
    mode=st.radio('Recommendation method',methods,horizontal=True)
    favorites=[]; uid=None; method='Item cosine'; alpha=0.6
    if mode!='Collaborative':
        favorites=st.multiselect('Search for movies you enjoy',list(labels),format_func=labels.get,default=[1] if 1 in labels else [])
    if mode!='Content-based':
        uid=st.selectbox('Select existing user ID',sorted(ratings.userId.unique()))
        method=st.selectbox('Collaborative algorithm',['Item cosine','Surprise SVD'])
    if mode=='Hybrid': alpha=st.slider('Content weight',0.0,1.0,0.6,0.05)
    if st.button('Generate recommendations',type='primary'):
        try:
            if mode=='Content-based': result=content.recommend(favorites,n)
            else:
                with st.spinner('Preparing recommendation model...'):
                    cf=cf_model(raw,method)
                    result=recommend_collaborative(content,cf,uid,n) if mode=='Collaborative' else recommend_hybrid(content,cf,uid,favorites,alpha,n)
            if result.empty: st.warning('No matching unseen movies with sufficient evidence. Try more favorites or a different user.')
            else:
                score_label={'Content-based':'Cosine similarity (0–1)','Collaborative':'Predicted rating (0.5–5)','Hybrid':'Combined score (0–1)'}[mode]
                st.caption(score_label+' · scores are not probabilities of liking a movie.')
                display=result.rename(columns={'movieId':'ID','title':'Movie','genres':'Genres','score':score_label})
                st.dataframe(display,hide_index=True,use_container_width=True,column_config={score_label:st.column_config.NumberColumn(format='%.3f')})
                st.download_button('Export recommendations',result.to_csv(index=False),'recommendations.csv','text/csv')
        except (ValueError,ImportError) as e: st.warning(str(e))
elif page=='Explore dataset':
    st.subheader('Exploratory data analysis')
    l,r=st.columns(2)
    with l: st.write('Movies by genre (multi-label)'); st.bar_chart(genre_counts(m))
    with r: st.write('Movies by release year'); st.line_chart(year_counts(m))
    st.dataframe(m,hide_index=True,use_container_width=True)
    if ratings is not None:
        st.subheader('User–item interactions'); st.json(interaction_summary(ratings,m))
        st.bar_chart(ratings.rating.value_counts().sort_index().rename('ratings'))
        st.write('Ratings per user'); st.bar_chart(ratings.groupby('userId').size().value_counts().sort_index().rename('users'))
    else: st.info('User–item interaction EDA needs real ratings.')
elif page=='Evaluate':
    st.subheader('Model evaluation')
    st.write('Metadata experiment: hide one genre from a movie query and check whether recommended movies contain it. This is a proxy, not measured user satisfaction.')
    if st.button('Run content evaluation'): st.dataframe(meta_eval(),hide_index=True)
    if ratings is not None:
        st.write('Item-cosine rating prediction: hold out one rating per eligible user; compare MAE/RMSE against a user-mean baseline. Cold items use the baseline and coverage is reported.')
        if st.button('Run ratings evaluation'):
            report=rate_eval(raw)
            if report is None: st.warning('Need at least one user with three distinct valid movie ratings.')
            else: st.json(report)
    else: st.info('Real ratings are needed to measure rating prediction errors. SVD and hybrid are not evaluated by this screen.')
else:
    st.subheader('How it works')
    st.markdown('**Content:** TF-IDF on complete genre tokens → normalized average favorites → cosine similarity.\n\n**Collaborative:** sparse user–item ratings → item cosine weighted rating prediction, or optional Surprise SVD.\n\n**Hybrid:** weighted blend of content similarity and scaled predicted rating, using candidates with collaborative evidence.\n\n**Limitations:** genre-only metadata, cold-start users/items, popularity bias, and no measured user preference accuracy from the bundled CSV.')
