import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express

st.set_page_config(page_title="Voter Fraud", page_icon=None, layout='centered', initial_sidebar_state='expanded', menu_items=None)

st.title('Voter Fraud <SEXY TITLE HERE>')

DATE_COLUMN = 'created_at'

@st.cache
def load_data():
    df_facebook_before_election = pd.read_csv('facebook_before_election.csv')
    df_facebook_after_election = pd.read_csv('facebook_after_election.csv')  
    df_reddit_before_election = pd.read_csv('reddit_before_election.csv')
    df_reddit_after_election = pd.read_csv('reddit_after_election.csv')
    df_twitter_before_election = pd.read_csv('twitter_before_election.csv')
    df_twitter_after_election = pd.read_csv('twitter_after_election.csv')

    df_facebook_before_insurrection = pd.read_csv('facebook_before_insurrection.csv')
    df_facebook_after_insurrection = pd.read_csv('facebook_after_insurrection.csv')  
    df_reddit_before_insurrection = pd.read_csv('reddit_before_insurrection.csv')
    df_reddit_after_insurrection = pd.read_csv('reddit_after_insurrection.csv')
    df_twitter_before_insurrection = pd.read_csv('twitter_before_insurrection.csv')
    df_twitter_after_insurrection = pd.read_csv('twitter_after_insurrection.csv')

    return df_facebook_before_election, df_facebook_after_election, 
    df_reddit_before_election, df_reddit_after_election,
    df_twitter_before_election, df_twitter_after_election,
    df_facebook_before_insurrection, df_facebook_after_insurrection,
    df_reddit_before_insurrection, df_reddit_after_insurrection,
    df_twitter_before_insurrection, df_twitter_after_insurrection

df_facebook_before_election, df_facebook_after_election, 
    df_reddit_before_election, df_reddit_after_election,
    df_twitter_before_election, df_twitter_after_election,
    df_facebook_before_insurrection, df_facebook_after_insurrection,
    df_reddit_before_insurrection, df_reddit_after_insurrection,
    df_twitter_before_insurrection, df_twitter_after_insurrection = load_data()

# General button colours
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #F3F3F3;
    color: #FFFFFF;
    border: 2px solid #F3F3F3;
    border-radius: 10px 10px 10px 10px;
}
div.stButton > button:focus {
    background-color: #E5E5E5;
    color: #FFFFFF;
    border: 2px solid #D9EAD3;
    border-radius: 10px 10px 10px 10px;
}
</style>""", unsafe_allow_html=True)

# Project description part 
desc_1, desc_space, desc_2 = st.columns(2, 0.1, 1)
with desc_1:
    st.markdown('<h3>Project Description<h3><p>Insert Text Here<p>', unsafe_allow_html=True)
with desc_2: 
    # TO GET ADYA IMAGE
    # st.image()

# Hashtags description
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    election_btn = st.button('<h3>US Elections 2020<h3><br><p>#corruptelection #electionfraud, #electionintegrity #fakeelection #fakevotes #voterfraud<p>', unsafe_allow_html=True, value=True)
with btn_col2:
    insurrection_btn = st.button('<h3>US Insurrection 2021<h3><br><p>#magacivilwar #marchfortrump #millionmagamarch #saveamerica #stopthesteal #stopthefraud<p>', unsafe_allow_html=True, value=False)

if election_btn:
    SELECTION = 'election'
    # Not sure whether to change the load_data function to just load those datasets or load everything
    df_facebook_before = df_facebook_before_election
    df_facebook_after = df_facebook_after_election
    df_reddit_before = df_reddit_before_election
    df_reddit_after = df_reddit_after_election
    df_twitter_after = df_twitter_after_election
    df_twitter_before = df_twitter_before_election
elif insurrection_btn:
    SELECTION = 'insurrection'
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

# Select Fb Reddit Twitter

# Map 

# Bar chart 

# Slider bar

# Before after bubbles

# Sentiment bar chart 

# Emotions bar chart 

# Emotion selection

# Boxplot

# Sample posts