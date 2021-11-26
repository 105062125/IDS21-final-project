import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express
import matplotlib.pyplot as plt
from plotly import graph_objs as go

st.set_page_config(page_title="Voter Fraud", page_icon=None, layout='centered', initial_sidebar_state='expanded', menu_items=None)

st.title('The United States Election and Insurrection')

DATE_COLUMN = 'created_at'

@st.cache
def load_data():
    df_facebook_before_election = pd.read_csv('./data/facebook_before_election.csv')
    df_facebook_after_election = pd.read_csv('./data/facebook_after_election.csv')  
    df_reddit_before_election = pd.read_csv('./data/reddit_before_election.csv')
    df_reddit_after_election = pd.read_csv('./data/reddit_after_election.csv')
    df_twitter_before_election = pd.read_csv('./data/twitter_before_election.csv')
    df_twitter_after_election = pd.read_csv('./data/twitter_after_election.csv')

    df_facebook_before_election[DATE_COLUMN] = df_facebook_before_election[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
    df_facebook_before_election[DATE_COLUMN] = pd.to_datetime(df_facebook_before_election[DATE_COLUMN])

    df_facebook_after_election[DATE_COLUMN] = df_facebook_after_election[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
    df_facebook_after_election[DATE_COLUMN] = pd.to_datetime(df_facebook_after_election[DATE_COLUMN])

    df_reddit_before_election[DATE_COLUMN] = df_reddit_before_election[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
    df_reddit_before_election[DATE_COLUMN] = pd.to_datetime(df_reddit_before_election[DATE_COLUMN])

    df_reddit_after_election[DATE_COLUMN] = df_reddit_after_election[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
    df_reddit_after_election[DATE_COLUMN] = pd.to_datetime(df_reddit_after_election[DATE_COLUMN])

    df_twitter_before_election[DATE_COLUMN] = df_twitter_before_election[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
    df_twitter_before_election[DATE_COLUMN] = pd.to_datetime(df_twitter_before_election[DATE_COLUMN])

    df_twitter_after_election[DATE_COLUMN] = df_twitter_after_election[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
    df_twitter_after_election[DATE_COLUMN] = pd.to_datetime(df_twitter_after_election[DATE_COLUMN])

    df_facebook_before_insurrection = pd.read_csv('./data/facebook_before_insurrection.csv')
    df_facebook_after_insurrection = pd.read_csv('./data/facebook_after_insurrection.csv')  
    df_reddit_before_insurrection = pd.read_csv('./data/reddit_before_insurrection.csv')
    df_reddit_after_insurrection = pd.read_csv('./data/reddit_after_insurrection.csv')
    df_twitter_before_insurrection = pd.read_csv('./data/twitter_before_insurrection.csv')
    df_twitter_after_insurrection = pd.read_csv('./data/twitter_after_insurrection.csv')

    df_facebook_before_insurrection[DATE_COLUMN] = df_facebook_before_insurrection[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
    df_facebook_before_insurrection[DATE_COLUMN] = pd.to_datetime(df_facebook_before_insurrection[DATE_COLUMN])

    df_facebook_after_insurrection[DATE_COLUMN] = df_facebook_after_insurrection[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
    df_facebook_after_insurrection[DATE_COLUMN] = pd.to_datetime(df_facebook_after_insurrection[DATE_COLUMN])

    df_reddit_before_insurrection[DATE_COLUMN] = df_reddit_before_insurrection[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
    df_reddit_before_insurrection[DATE_COLUMN] = pd.to_datetime(df_reddit_before_insurrection[DATE_COLUMN])

    df_reddit_after_insurrection[DATE_COLUMN] = df_reddit_after_insurrection[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
    df_reddit_after_insurrection[DATE_COLUMN] = pd.to_datetime(df_reddit_after_insurrection[DATE_COLUMN])

    df_twitter_before_insurrection[DATE_COLUMN] = df_twitter_before_insurrection[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
    df_twitter_before_insurrection[DATE_COLUMN] = pd.to_datetime(df_twitter_before_insurrection[DATE_COLUMN])

    df_twitter_after_insurrection[DATE_COLUMN] = df_twitter_after_insurrection[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
    df_twitter_after_insurrection[DATE_COLUMN] = pd.to_datetime(df_twitter_after_insurrection[DATE_COLUMN])

    return df_facebook_before_election, df_facebook_after_election, df_reddit_before_election, df_reddit_after_election, df_twitter_before_election, df_twitter_after_election,df_facebook_before_insurrection, df_facebook_after_insurrection,df_reddit_before_insurrection, df_reddit_after_insurrection, df_twitter_before_insurrection, df_twitter_after_insurrection

df_facebook_before_election, df_facebook_after_election, df_reddit_before_election, df_reddit_after_election, df_twitter_before_election, df_twitter_after_election, df_facebook_before_insurrection, df_facebook_after_insurrection, df_reddit_before_insurrection, df_reddit_after_insurrection, df_twitter_before_insurrection, df_twitter_after_insurrection = load_data()

# General button colours
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #F3F3F3;
    color: black;
    border: 2px solid #F3F3F3;
    border-radius: 10px 10px 10px 10px;
}
div.stButton > button:focus {
    background-color: #E5E5E5;
    color: black;
    border: 2px solid #D9EAD3;
    border-radius: 10px 10px 10px 10px;
}
div.row-widget.stRadio > div {
    flex-direction:row;
    justify-content: space-around;
    background-color: #E5E5E5;
    padding-top: 20px;
    padding-bottom: 20px;
    border-radius: 10px 10px 10px 10px;
}
</style>""", unsafe_allow_html=True)

# Project description part 
desc_1, desc_space, desc_2 = st.columns((2, 0.1, 1))
with desc_1:
    st.markdown('<h3>Project Description<h3><p>The 2020 United States presidential election was held on 3 November 2020. The Democratic nominee Joe Biden won the election. Before, during and after Election Day, Republicans attempted to overturn the results, by calling out widespread voter fraud. On January 6, 2021, a mob of supporters of then-President Trump attacked the United States Capitol, seeking to overturn the Congress session that would formalize the Democrat victory.<p>', unsafe_allow_html=True)
with desc_2: 
    st.image('images/img.png')

# Hashtags description
# Can't figure out how to make the button CSS
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    election_btn = st.button('<h3>US Elections 2020<h3><br><p>#corruptelection #electionfraud, #electionintegrity #fakeelection #fakevotes #voterfraud<p>')
with btn_col2:
    insurrection_btn = st.button('<h3>US Insurrection 2021<h3><br><p>#magacivilwar #marchfortrump #millionmagamarch #saveamerica #stopthesteal #stopthefraud<p>')

election_btn = True

if election_btn:
    SELECTION = 'election'
    if 'selection' not in st.session_state:
        st.session_state.selection = 'election'

    # Not sure whether to change the load_data function to just load those datasets or load everything
    df_facebook_before = df_facebook_before_election
    df_facebook_after = df_facebook_after_election
    df_reddit_before = df_reddit_before_election
    df_reddit_after = df_reddit_after_election
    df_twitter_after = df_twitter_after_election
    df_twitter_before = df_twitter_before_election
elif insurrection_btn:
    SELECTION = 'insurrection'
    if 'selection' not in st.session_state:
        st.session_state.selection = 'insurrection'

    df_facebook_before = df_facebook_before_insurrection
    df_facebook_after = df_facebook_after_insurrection
    df_reddit_before = df_reddit_before_insurrection
    df_reddit_after = df_reddit_after_insurrection
    df_twitter_after = df_twitter_after_insurrection
    df_twitter_before = df_twitter_before_insurrection

# Statistics of each social media platform
stat_col1, stat_col2, stat_col3 = st.columns(3)

facebook_total = len(df_facebook_before) + len(df_facebook_after)
reddit_total = len(df_reddit_before) + len(df_reddit_after)
twitter_total = len(df_twitter_before) + len(df_twitter_after)
stat_col1.metric(label = "Facebook", value=f"{facebook_total:,}")
stat_col2.metric(label = "Reddit", value=f"{reddit_total:,}")
stat_col3.metric(label = "Twitter", value=f"{twitter_total:,}")

stat_breakdown_col1, stat_breakdown_col2, stat_breakdown_col3 = st.columns(3)

# Line chart 

st.markdown('Talk about numbers and peaks etc')

# Select Fb Reddit Twitter
social_media_selector = st.radio("", ('Facebook', 'Twitter', 'Reddit'))
if 'social_media' not in st.session_state:
    st.session_state.social_media = social_media_selector

# Map 

# Bar chart 

# Slider bar
df_all_before = pd.concat([df_twitter_before, df_facebook_before, df_reddit_before])
df_all_after = pd.concat([df_twitter_after, df_facebook_after, df_reddit_after])

min_date = min(df_all_before[DATE_COLUMN]).date()
max_date = max(df_all_after[DATE_COLUMN]).date()
date_format = 'YYYY-MM-DD'

date_filter = st.slider('Select date range', min_value=min_date, max_value=max_date, value=(min_date, max_date), format=date_format)

# Before after bubbles

# Emotions bar chart 
st.markdown('<h4>Emotions in Discourse<h4>', unsafe_allow_html=True)
st.markdown('<p style=color:grey;font-size:1em;>Sentiment<p>', unsafe_allow_html=True)

st.markdown('<p style=color:grey;font-size:1em;>Emotion<p>', unsafe_allow_html=True)


# Emotion selection
st.markdown('<p style=font-weight:bold;font-size:1rem;color:grey;>EMOTIONS OVER TIME <p>', unsafe_allow_html=True)
st.markdown('TO DO: Explain shares of emotions that are dominating over time ')
emotion_selector = st.radio("", ('Sadness', 'Anger', 'Disgust', 'Fear', 'Joy'))
if 'emotion' not in st.session_state:
    st.session_state.emotion = emotion_selector.lower()

emotion_1, emotion_space, emotion_2 = st.columns((2, 0.1, 1))

# Boxplot
with emotion_1:
    st.write('Box plot of ' + emotion_selector + ' over time ')

# Sample posts
with emotion_2:
    st.markdown('<p style=color:grey;font-weight:bold;>' + emotion_selector + '<p>', unsafe_allow_html=True)
    st.markdown('<p style=color:grey;>Sample post before ' + st.session_state.selection + '<p>', unsafe_allow_html=True)
    if st.session_state.social_media == 'Facebook':
        filtered_posts_before = df_facebook_before[df_facebook_before['highest_emotion'] == st.session_state.emotion]['text'].sample().values[0]
    elif st.session_state.social_media == 'Reddit':
        filtered_posts_before = df_reddit_before[df_reddit_before['highest_emotion'] == st.session_state.emotion]['text'].sample().values[0]
    elif st.session_state.social_media == 'Twitter':
        filtered_posts_before = df_twitter_before[df_twitter_before['highest_emotion'] == st.session_state.emotion]['text'].sample().values[0]

    filtered_posts_before_truncated = (filtered_posts_before[:300] + '..') if len(filtered_posts_before) > 300 else filtered_posts_before
    st.markdown('<p style=background-color:#CCCCCC;padding:8px;border-radius:3px;>' + filtered_posts_before_truncated + '<p>', unsafe_allow_html=True)
    
    st.markdown('<p style=color:grey;>Sample post after ' + st.session_state.selection + '<p>', unsafe_allow_html=True)

    if st.session_state.social_media == 'Facebook':
        filtered_posts_after = df_facebook_after[df_facebook_after['highest_emotion'] == st.session_state.emotion]['text'].sample().values[0]
    elif st.session_state.social_media == 'Reddit':
        filtered_posts_after = df_reddit_after[df_reddit_after['highest_emotion'] == st.session_state.emotion]['text'].sample().values[0]
    elif st.session_state.social_media == 'Twitter':
        filtered_posts_after = df_twitter_after[df_twitter_after['highest_emotion'] == st.session_state.emotion]['text'].sample().values[0]
    filtered_posts_after_truncated = (filtered_posts_after[:300] + '..') if len(filtered_posts_after) > 300 else filtered_posts_after
    st.markdown('<p style=background-color:#CCCCCC;padding:8px;border-radius:3px;>' + filtered_posts_after_truncated + '<p>', unsafe_allow_html=True)

