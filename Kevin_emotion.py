from numpy.core.fromnumeric import sort
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.express
import matplotlib.pyplot as plt
from plotly import graph_objs as go
# from wordcloud import WordCloud, STOPWORDS
import re
# import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px 



st.set_page_config(page_title="US Election and Insurrection", page_icon=None, layout='wide', initial_sidebar_state='expanded', menu_items=None)

st.title('The United States Election and Insurrection')

DATE_COLUMN = 'created_at'


@st.cache
def load_data():

    df_facebook_before_insurrection = pd.read_csv('./data/facebook_before_insurrection.csv')
    df_facebook_after_insurrection = pd.read_csv('./data/facebook_after_insurrection.csv')
    df_reddit_before_insurrection = pd.read_csv('./data/reddit_before_insurrection.csv')
    df_reddit_after_insurrection = pd.read_csv('./data/reddit_after_insurrection.csv')
    df_twitter_before_insurrection = pd.read_csv('./data/twitter_before_insurrection.csv')
    df_twitter_after_insurrection = pd.read_csv('./data/twitter_after_insurrection.csv')
    df_insurrection_before_all = pd.concat([df_facebook_before_insurrection, df_reddit_before_insurrection, df_twitter_before_insurrection])
    df_insurrection_after_all = pd.concat([df_facebook_after_insurrection, df_reddit_after_insurrection, df_twitter_after_insurrection])
    df_facebook_all = pd.concat([df_facebook_before_insurrection, df_facebook_after_insurrection])
    df_facebook_all['date_column'] = pd.to_datetime(df_facebook_all['created_at']).dt.date
    df_reddit_all = pd.concat([df_reddit_before_insurrection, df_reddit_after_insurrection])
    df_reddit_all['date_column'] = pd.to_datetime(df_reddit_all['created_at']).dt.date
    df_twitter_all = pd.concat([df_twitter_before_insurrection, df_twitter_after_insurrection])
    df_twitter_all['date_column'] = pd.to_datetime(df_twitter_all['created_at']).dt.date
    df_all = pd.concat([df_insurrection_before_all, df_insurrection_after_all])

    return df_facebook_all, df_reddit_all, df_twitter_all, df_all, df_insurrection_before_all, df_insurrection_after_all, df_facebook_before_insurrection, df_facebook_after_insurrection, df_reddit_before_insurrection, df_reddit_after_insurrection, df_twitter_before_insurrection, df_twitter_after_insurrection

@st.cache
def load_data2():

    df_facebook_before_election = pd.read_csv('./data/facebook_before_election.csv')
    df_facebook_after_election = pd.read_csv('./data/facebook_after_election.csv')
    df_reddit_before_election = pd.read_csv('./data/reddit_before_election.csv')
    df_reddit_after_election = pd.read_csv('./data/reddit_after_election.csv')
    df_twitter_before_election = pd.read_csv('./data/twitter_before_election.csv')
    df_twitter_after_election = pd.read_csv('./data/twitter_after_election.csv')
    df_election_before_all = pd.concat([df_facebook_before_election, df_reddit_before_election, df_twitter_before_election])
    df_election_after_all = pd.concat([df_facebook_after_election, df_reddit_after_election, df_twitter_after_election])
    df_facebook_all_election = pd.concat([df_facebook_before_election, df_facebook_after_election])
    df_facebook_all_election['date_column'] = pd.to_datetime(df_facebook_all_election['created_at']).dt.date
    df_reddit_all_election = pd.concat([df_reddit_before_election, df_reddit_after_election])
    df_reddit_all_election['date_column'] = pd.to_datetime(df_reddit_all_election['created_at']).dt.date
    df_twitter_all_election = pd.concat([df_twitter_before_election, df_twitter_after_election])
    df_twitter_all_election['date_column'] = pd.to_datetime(df_twitter_all_election['created_at']).dt.date
    df_all_election = pd.concat([df_election_before_all, df_election_after_all])

    return df_facebook_all_election, df_reddit_all_election, df_twitter_all_election, df_all_election


#df_linechart_freq = {fb_ctdate[], fb_ctdate['id'], tw_ctdate['id'], rd_ctdate['id']}

#st.table(df_linechart_freq)

# df_tw_geo_before_insurrection = pd.read_csv('./data/before_with_places_ext.csv')
# df_tw_geo_after_insurrection = pd.read_csv('./data/after_with_places_ext.csv')


# Project description part
desc_1, desc_space, desc_2 = st.columns((2, 0.1, 1))
with desc_1:
    st.markdown('<h3>Project Description<h3><p>The 2020 United States presidential election was held on 3 November 2020. The Democratic nominee Joe Biden won the election. Before, during and after Election Day, Republicans attempted to overturn the results, by calling out widespread voter fraud. On January 6, 2021, a mob of supporters of then-President Trump attacked the United States Capitol, seeking to overturn the Congress session that would formalize the Democrat victory.<p>', unsafe_allow_html=True)
with desc_2:
    st.image('images/img.png')

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    election_btn = st.button('<h3>US Elections 2020<h3><br><p>#corruptelection #electionfraud, #electionintegrity #fakeelection #fakevotes #voterfraud<p>')
with btn_col2:
    insurrection_btn = st.button('<h3>US Insurrection 2021<h3><br><p>#magacivilwar #marchfortrump #millionmagamarch #saveamerica #stopthesteal #stopthefraud<p>')

# Statistics of each social media platform
stat_col1, stat_col2, stat_col3 = st.columns(3)

df_facebook_all, df_reddit_all, df_twitter_all, df_all, df_insurrection_before_all, df_insurrection_after_all, df_facebook_before_insurrection, df_facebook_after_insurrection, df_reddit_before_insurrection, df_reddit_after_insurrection, df_twitter_before_insurrection, df_twitter_after_insurrection = load_data()
df_facebook_all_election, df_reddit_all_election, df_twitter_all_election, df_all_election = load_data2()

df_facebook_before = df_facebook_before_insurrection
df_facebook_after = df_facebook_after_insurrection
df_reddit_before = df_reddit_before_insurrection
df_reddit_after = df_reddit_after_insurrection
df_twitter_after = df_twitter_after_insurrection
df_twitter_before = df_twitter_before_insurrection

# df_facebook_all = pd.concat([df_facebook_before_insurrection, df_facebook_after_insurrection])
# df_reddit_all = pd.concat([df_reddit_before_insurrection, df_reddit_after_insurrection])
# df_twitter_all = pd.concat([df_twitter_before_insurrection, df_twitter_after_insurrection])


# METRICS
facebook_total = len(df_facebook_before) + len(df_facebook_after)
reddit_total = len(df_reddit_before) + len(df_reddit_after)
twitter_total = len(df_twitter_before) + len(df_twitter_after)
stat_col1.metric(label = "Facebook", value=f"{facebook_total:,}")
stat_col2.metric(label = "Reddit", value=f"{reddit_total:,}")
stat_col3.metric(label = "Twitter", value=f"{twitter_total:,}")

stat_breakdown_col1, stat_breakdown_col2, stat_breakdown_col3, stat_breakdown_col4, stat_breakdown_col5, stat_breakdown_col6 = st.columns(6)
stat_breakdown_col1.metric(label="BEFORE", value=f"{len(df_facebook_before):,}")
stat_breakdown_col2.metric(label="AFTER", value=f"{len(df_facebook_after):,}")

stat_breakdown_col3.metric(label="BEFORE", value=f"{len(df_reddit_before):,}")
stat_breakdown_col4.metric(label="AFTER", value=f"{len(df_reddit_after):,}")

stat_breakdown_col5.metric(label="BEFORE", value=f"{len(df_twitter_before):,}")
stat_breakdown_col6.metric(label="AFTER", value=f"{len(df_twitter_after):,}")

# LINECHART
df_fb_count = pd.concat([df_facebook_before_insurrection[['id','created_at']],df_facebook_after_insurrection[['id','created_at']]])
df_rd_count = pd.concat([df_reddit_before_insurrection[['id','created_at']],df_reddit_after_insurrection[['id','created_at']]])
df_tw_count = pd.concat([df_twitter_before_insurrection[['id','created_at']],df_twitter_after_insurrection[['id','created_at']]])

df_fb_count[DATE_COLUMN] = df_fb_count[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
df_fb_count[DATE_COLUMN] = pd.to_datetime(df_fb_count[DATE_COLUMN])

df_rd_count[DATE_COLUMN] = df_rd_count[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
df_rd_count[DATE_COLUMN] = pd.to_datetime(df_rd_count[DATE_COLUMN])

df_tw_count[DATE_COLUMN] = df_tw_count[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
df_tw_count[DATE_COLUMN] = pd.to_datetime(df_tw_count[DATE_COLUMN])

fb_ctdate = df_fb_count.groupby([df_fb_count[DATE_COLUMN].dt.date]).count().drop(columns=DATE_COLUMN).reset_index()
tw_ctdate = df_tw_count.groupby([df_tw_count[DATE_COLUMN].dt.date]).count().drop(columns=DATE_COLUMN).reset_index()
rd_ctdate = df_rd_count.groupby([df_rd_count[DATE_COLUMN].dt.date]).count().drop(columns=DATE_COLUMN).reset_index()

data_linechart = {'date':fb_ctdate['created_at'],'Facebook':fb_ctdate['id'],'Twitter':tw_ctdate['id'],'Reddit':rd_ctdate['id']}
df_linechart = pd.DataFrame(data=data_linechart)
df_linechart.set_index(['date'], inplace=True)

st.line_chart(df_linechart)

# LINECHART DESCRIPTOR

st.markdown('Talk about numbers, then which source generated the most content bla bla bla, Facebook peaks where, and desc of trend Twitter peaks where and desc of trend Reddit peaks where')

# SLIDER

min_date = min(df_linechart.index)
max_date = max(df_linechart.index)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{padding:0 20px;}</style>', unsafe_allow_html=True)

st.write('<style>div[role=radiogroup]{background-color:#eaeaea;border-radius:10px;padding:10px}</style>', unsafe_allow_html=True)


choose=st.radio("Social media filter",("Facebook","Reddit","Twitter"))

date_format = 'YYYY-MM-DD'


geo_after_df = pd.read_csv('./data/after_with_places_ext.csv')
geo_before_df = pd.read_csv('./data/before_with_places_ext.csv')

st.markdown("""---""")

time_container = st.container()
time_container.markdown('## Discourse over time')
time_container.markdown('Describe what you can do in this time-based ')

date_filter = time_container.slider('Select date range', min_value=min_date, max_value=max_date, value=(min_date, max_date), format=date_format)

geo_all=(pd.concat([geo_before_df,geo_after_df]))
geo_all = geo_all[geo_all['country_code']=='US']



us_state_to_abbrev = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}

inverted_us_state = dict(map(reversed, us_state_to_abbrev.items()))

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')

for col in df.columns:
    df[col] = df[col].astype(str)

df['text'] = df['state'] + '<br>' + \
    'Beef ' + df['beef'] + ' Dairy ' + df['dairy'] + '<br>' + \
    'Fruits ' + df['total fruits'] + ' Veggies ' + df['total veggies'] + '<br>' + \
    'Wheat ' + df['wheat'] + ' Corn ' + df['corn']


df_twitter_after_insurrection = pd.read_csv('./data/twitter_after_insurrection.csv')


fig = go.Figure(data=go.Choropleth(
    locations=df['code'],
    z=df['total exports'].astype(float),
    locationmode='USA-states',
    colorscale='Reds',
    autocolorscale=False,
    text=df['text'], # hover text
    marker_line_color='white', # line markers between states
    colorbar_title="Millions USD"
))

fig.update_layout(
    title_text='Discourse in US States',
    geo = dict(
        scope='usa',
        projection=go.layout.geo.Projection(type = 'albers usa'),
        showlakes=True, # lakes
        lakecolor='rgb(255, 255, 255)'),
)

time_container.plotly_chart(fig)

st.markdown("""---""")

emotion_container = st.container()
emotion_container.markdown('## Emotions in discourse')
selector_way = st.radio(" ", ('Election', 'Insurrection'))
social_selector_emotion = st.radio(" ", ('Facebook', 'Reddit', 'Twitter'))
st.markdown('<p style=color:grey;font-size:1em;>Sentiment<p>', unsafe_allow_html=True)
box_posneg_1, box_emotion_up_space, box_posneg_2 = st.columns((2, 0.1, 1))
if selector_way == 'Insurrection':
    with box_posneg_1:
        st.markdown('TO DO: Explain what\'s going on in the boxplot Insurrection')
    with box_posneg_2:
        st.markdown('TO DO: Explain what\'s going on in the boxplot Insurrection')
elif selector_way == 'Election':
    with box_posneg_1:
        st.markdown('TO DO: Explain what\'s going on in the boxplot Election')
    with box_posneg_2:
        st.markdown('TO DO: Explain what\'s going on in the boxplot Election')



st.markdown('<p style=color:grey;font-size:1em;>Emotion over time<p>', unsafe_allow_html=True)

box_emotion_up_1, box_emotion_up_space, box_emotion_up_2, box_emotion_up_3 = st.columns((2, 0.1, 2, 1))

if selector_way == 'Insurrection':
    with box_emotion_up_1:
        if social_selector_emotion == 'Facebook':
            df_stack = pd.DataFrame({'sadness': df_facebook_all['emotion.sadness'].tolist(),'anger': df_facebook_all['emotion.anger'].tolist(),'disgust': df_facebook_all['emotion.disgust'].tolist(),'joy': df_facebook_all['emotion.joy'].tolist(),'fear': df_facebook_all['emotion.fear'].tolist(), 'date':df_facebook_all['date_column'].tolist() })
            df_stack = df_stack.groupby('date').agg('sum')
            # fig = px.bar(df_stack, x="date", y=["sadness", "anger", "disgust"])
            # st.write(fig)
            st.bar_chart(df_stack)
        elif social_selector_emotion == 'Reddit':
            df_stack = pd.DataFrame({'sadness': df_reddit_all['emotion.sadness'].tolist(),'anger': df_reddit_all['emotion.anger'].tolist(),'disgust': df_reddit_all['emotion.disgust'].tolist(),'joy': df_reddit_all['emotion.joy'].tolist(),'fear': df_reddit_all['emotion.fear'].tolist(), 'date':df_reddit_all['date_column'].tolist() })
            df_stack = df_stack.groupby('date').agg('sum')
            st.bar_chart(df_stack)
        elif social_selector_emotion == 'Twitter':
            df_stack = pd.DataFrame({'sadness': df_twitter_all['emotion.sadness'].tolist(),'anger': df_twitter_all['emotion.anger'].tolist(),'disgust': df_twitter_all['emotion.disgust'].tolist(),'joy': df_twitter_all['emotion.joy'].tolist(),'fear': df_twitter_all['emotion.fear'].tolist(), 'date':df_twitter_all['date_column'].tolist() })
            df_stack = df_stack.groupby('date').agg('sum')
            st.bar_chart(df_stack)


    with box_emotion_up_2:
        
        if social_selector_emotion == 'Facebook':
            df_emo = pd.DataFrame(df_facebook_all['highest_emotion'].value_counts())
            # fig=px.bar(df_emo, orientation='h')
            # st.write(fig)
            st.bar_chart(df_emo)
        elif social_selector_emotion == 'Reddit':
            df_emo = pd.DataFrame(df_reddit_all['highest_emotion'].value_counts())
            # fig=px.bar(df_emo, orientation='h')
            # st.write(fig)
            st.bar_chart(df_emo)
        elif social_selector_emotion == 'Twitter':
            df_emo = pd.DataFrame(df_twitter_all['highest_emotion'].value_counts())
            # fig=px.bar(df_emo, orientation='h')
            # st.write(fig)
            st.bar_chart(df_emo)
    with box_emotion_up_3:
        st.markdown('TO DO: Explain what\'s going on in the boxplot')

elif selector_way == 'Election':
    with box_emotion_up_1:
        if social_selector_emotion == 'Facebook':
            df_stack = pd.DataFrame({'sadness': df_facebook_all_election['emotion.sadness'].tolist(),'anger': df_facebook_all_election['emotion.anger'].tolist(),'disgust': df_facebook_all_election['emotion.disgust'].tolist(),'joy': df_facebook_all_election['emotion.joy'].tolist(),'fear': df_facebook_all_election['emotion.fear'].tolist(), 'date':df_facebook_all_election['date_column'].tolist() })
            df_stack = df_stack.groupby('date').agg('sum')
            # fig = px.bar(df_stack, x="date", y=["sadness", "anger", "disgust"])
            # st.write(fig)
            st.bar_chart(df_stack)
        elif social_selector_emotion == 'Reddit':
            df_stack = pd.DataFrame({'sadness': df_reddit_all_election['emotion.sadness'].tolist(),'anger': df_reddit_all_election['emotion.anger'].tolist(),'disgust': df_reddit_all_election['emotion.disgust'].tolist(),'joy': df_reddit_all_election['emotion.joy'].tolist(),'fear': df_reddit_all_election['emotion.fear'].tolist(), 'date':df_reddit_all_election['date_column'].tolist() })
            df_stack = df_stack.groupby('date').agg('sum')
            st.bar_chart(df_stack)
        elif social_selector_emotion == 'Twitter':
            df_stack = pd.DataFrame({'sadness': df_twitter_all_election['emotion.sadness'].tolist(),'anger': df_twitter_all_election['emotion.anger'].tolist(),'disgust': df_twitter_all_election['emotion.disgust'].tolist(),'joy': df_twitter_all_election['emotion.joy'].tolist(),'fear': df_twitter_all_election['emotion.fear'].tolist(), 'date':df_twitter_all_election['date_column'].tolist() })
            df_stack = df_stack.groupby('date').agg('sum')
            st.bar_chart(df_stack)


    with box_emotion_up_2:
        
        if social_selector_emotion == 'Facebook':
            df_emo = pd.DataFrame(df_facebook_all_election['highest_emotion'].value_counts())
            # fig=px.bar(df_emo, orientation='h')
            # st.write(fig)
            st.bar_chart(df_emo)
        elif social_selector_emotion == 'Reddit':
            df_emo = pd.DataFrame(df_reddit_all_election['highest_emotion'].value_counts())
            # fig=px.bar(df_emo, orientation='h')
            # st.write(fig)
            st.bar_chart(df_emo)
        elif social_selector_emotion == 'Twitter':
            df_emo = pd.DataFrame(df_twitter_all_election['highest_emotion'].value_counts())
            # fig=px.bar(df_emo, orientation='h')
            # st.write(fig)
            st.bar_chart(df_emo)
    with box_emotion_up_3:
        st.markdown('TO DO: Explain what\'s going on in the boxplot')






    # df_stack.iloc[:100].plot.bar(stacked=True)
# df_stack = pd.DataFrame({'sadness': df_all['emotion.sadness'].tolist(),'anger': df_all['emotion.anger'].tolist(),'disgust': df_all['emotion.disgust'].tolist(),'joy': df_all['emotion.joy'].tolist(),'fear': df_all['emotion.fear'].tolist(), 'date':df_all['date_column'].tolist() })
# df_stack = df_stack.groupby('date').agg('sum')
# df_stack.plot.bar(stacked=True)



st.markdown('<p style=font-weight:bold;font-size:1rem;color:grey;>EMOTIONS OVER TIME <p>', unsafe_allow_html=True)
st.markdown('TO DO: Explain shares of emotions that are dominating over time ')
ways = st.radio("", ('Election', 'Insurrection'))
social_selector = st.radio("", ('All', 'Facebook', 'Reddit', 'Twitter'))
emotion_selector = st.radio("", ('Sadness', 'Anger', 'Disgust', 'Fear', 'Joy'))
if 'emotion' not in st.session_state:
    st.session_state.emotion = emotion_selector.lower()

box_emotion_1, box_emotion_space, box_emotion_2 = st.columns((2, 0.1, 1))

df_insurrection_before_all_len = len(df_insurrection_before_all)
df_insurrection_after_all_len = len(df_insurrection_after_all)

with box_emotion_1:
    st.write('Box plot of ' + emotion_selector + ' in '+ social_selector + ' '+ ways + ' data over time ')
    if social_selector == 'Facebook': social_selector='fb'
    elif social_selector == 'Reddit': social_selector='rd'
    elif social_selector == 'Twitter': social_selector='tw'
    elif social_selector == 'All': social_selector='all'
    st.image('images/{}_{}_{}.png'.format(social_selector, ways.lower(), emotion_selector.lower()))

with box_emotion_2:
    st.markdown('<p style=font-weight:bold;font-size:3rem;color:grey;>{}<p>'.format(emotion_selector.upper()), unsafe_allow_html=True)
    st.markdown('TO DO: Explain what\'s going on in the boxplot')
    st.markdown('<p style=font-weight:bold;font-size:1rem;color:grey;> Ramdomly pick post with {} emotion before {} <p>'.format(emotion_selector.upper(), ways.lower()), unsafe_allow_html=True)
    post_before = df_insurrection_before_all[df_insurrection_before_all['highest_emotion']==emotion_selector.lower()].sample()
    post_before_text = post_before['text'].values[0]
    post_before_user = post_before['author_id'].values[0]
    st.markdown(post_before_user + ':')
    st.markdown(post_before_text)
    st.markdown('<p style=font-weight:bold;font-size:1rem;color:grey;> Ramdomly pick post with {} emotion after {} <p>'.format(emotion_selector.upper(), ways.lower()), unsafe_allow_html=True)
    post_after = df_insurrection_after_all[df_insurrection_after_all['highest_emotion']==emotion_selector.lower()].sample()
    post_after_text = post_after['text'].values[0]
    post_after_user = post_after['author_id'].values[0]
    st.markdown(post_after_user + ':')
    st.markdown(post_after_text)
    # if social_selector == 'all' and emotion_selector.lower() == 'sadness':

    
    

