from numpy.core.fromnumeric import sort
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.express
import matplotlib.pyplot as plt
from plotly import graph_objs as go
from wordcloud import WordCloud, STOPWORDS
import re
import pickle

# import pydeck as pdk
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import base64

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

st.set_page_config(page_title="Exploring Online Discourse on Voter Fraud Allegations in the 2020 US Elections and the January 6 Insurrection", page_icon=None, layout='centered', initial_sidebar_state='expanded', menu_items=None)

header_html = "<img src='data:image/png;base64,{}' width=100%>".format(
    img_to_bytes("header.png")
)
st.markdown(
    header_html, unsafe_allow_html=True,
)


st.title('What Happened Then?')
st.markdown('<h3  style = "color: #aaa">Exploring Online Discourse on Voter Fraud Allegations in the 2020 US Elections and the January 6 Insurrection</h3>', unsafe_allow_html=True)
DATE_COLUMN = 'created_at'


@st.cache(allow_output_mutation=True)
def load_data():

    with open('./data/facebook_before_insurrection.pkl', 'rb') as f:
        df_facebook_before_insurrection = pickle.load(f)
    with open('./data/facebook_after_insurrection.pkl', 'rb') as f:
        df_facebook_after_insurrection = pickle.load(f)
    with open('./data/reddit_before_insurrection.pkl', 'rb') as f:
        df_reddit_before_insurrection = pickle.load(f)
    with open('./data/reddit_after_insurrection.pkl', 'rb') as f:
        df_reddit_after_insurrection = pickle.load(f)
    with open('./data/twitter_before_insurrection.pkl', 'rb') as f:
        df_twitter_before_insurrection = pickle.load(f)
    with open('./data/twitter_after_insurrection.pkl', 'rb') as f:
        df_twitter_after_insurrection = pickle.load(f)

    df_facebook_all = pd.concat([df_facebook_before_insurrection, df_facebook_after_insurrection])
    df_facebook_all['date_column'] = pd.to_datetime(df_facebook_all['created_at']).dt.date
    df_reddit_all = pd.concat([df_reddit_before_insurrection, df_reddit_after_insurrection])
    df_reddit_all['date_column'] = pd.to_datetime(df_reddit_all['created_at']).dt.date
    df_twitter_all = pd.concat([df_twitter_before_insurrection, df_twitter_after_insurrection])
    df_twitter_all['date_column'] = pd.to_datetime(df_twitter_all['created_at']).dt.date

    df_insurrection_before_all = pd.concat([df_facebook_before_insurrection, df_reddit_before_insurrection, df_twitter_before_insurrection])
    df_insurrection_after_all = pd.concat([df_facebook_after_insurrection, df_reddit_after_insurrection, df_twitter_after_insurrection])

    df_all = pd.concat([df_facebook_all, df_reddit_all, df_twitter_all])

    return df_facebook_all, df_reddit_all, df_twitter_all, df_all, df_insurrection_before_all, df_insurrection_after_all, df_facebook_before_insurrection, df_facebook_after_insurrection, df_reddit_before_insurrection, df_reddit_after_insurrection, df_twitter_before_insurrection, df_twitter_after_insurrection

@st.cache
def load_data2():

    with open('./data/facebook_before_election.pkl', 'rb') as f:
        df_facebook_before_election = pickle.load(f)
    with open('./data/facebook_after_election.pkl', 'rb') as f:
        df_facebook_after_election = pickle.load(f)
    with open('./data/reddit_before_election.pkl', 'rb') as f:
        df_reddit_before_election = pickle.load(f)
    with open('./data/reddit_after_election.pkl', 'rb') as f:
        df_reddit_after_election = pickle.load(f)
    with open('./data/twitter_before_election.pkl', 'rb') as f:
        df_twitter_before_election = pickle.load(f)
    with open('./data/twitter_after_election.pkl', 'rb') as f:
        df_twitter_after_election = pickle.load(f)

    df_election_before_all = pd.concat([df_facebook_before_election, df_reddit_before_election, df_twitter_before_election])
    df_election_after_all = pd.concat([df_facebook_after_election, df_reddit_after_election, df_twitter_after_election])
    df_facebook_all_election = pd.concat([df_facebook_before_election, df_facebook_after_election])
    df_facebook_all_election['date_column'] = pd.to_datetime(df_facebook_all_election['created_at']).dt.date
    df_reddit_all_election = pd.concat([df_reddit_before_election, df_reddit_after_election])
    df_reddit_all_election['date_column'] = pd.to_datetime(df_reddit_all_election['created_at']).dt.date
    df_twitter_all_election = pd.concat([df_twitter_before_election, df_twitter_after_election])
    df_twitter_all_election['date_column'] = pd.to_datetime(df_twitter_all_election['created_at']).dt.date
    df_all_election = pd.concat([df_election_before_all, df_election_after_all])

    return df_facebook_before_election, df_facebook_after_election, df_reddit_before_election, df_reddit_after_election, df_twitter_before_election, df_twitter_after_election, df_facebook_all_election, df_reddit_all_election, df_twitter_all_election, df_all_election


#df_linechart_freq = {fb_ctdate[], fb_ctdate['id'], tw_ctdate['id'], rd_ctdate['id']}

#st.table(df_linechart_freq)

# df_tw_geo_before_insurrection = pd.read_csv('./data/before_with_places_ext.csv')
# df_tw_geo_after_insurrection = pd.read_csv('./data/after_with_places_ext.csv')


# Project description part

st.markdown('<p>This project aims to analyze the trend of discourse and emotional sentiment in social media around two topics in recent American politics: the voter fraud allegation around the 2020 presidential election and the attempt to take over Capitol Hill over the same voter fraud narrative.</p>', unsafe_allow_html=True)

#desc_1, desc_space, desc_2 = st.columns((2, 0.1, 1))
#with desc_1:
#with desc_2:
#    st.markdown('')

st.write('<style> span.caption{color: #aaa; font-size:rem1.5; text-transform: uppercase} .story-container{background: #eaeaea; padding: 10px; border: 1px solid #eaeaea; border-radius: 10px; margin-bottom: 10px} </style>', unsafe_allow_html=True)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    election_btn = st.write('<div class = "story-container"><h3>Voter Fraud in US Elections 2020</h3><p>The 2020 United States presidential election was held on 3 November 2020. The Democratic nominee Joe Biden won the election. Before, during and after Election Day, some groups attempted to overturn the results, by calling out widespread voter fraud</p><span class = "caption">Hashtags used to capture this phenomenon</span><p>#corruptelection #electionfraud, #electionintegrity #fakeelection #fakevotes #voterfraud<p></div>', unsafe_allow_html=True)
with btn_col2:
    insurrection_btn = st.write('<div class = "story-container"><h3>2021 United States Capitol Insurrection</h3><p>On January 6, 2021, a mob of supporters of then-President Trump attacked the United States Capitol after a rally the president held, seeking to overturn the Congress session that would formalize the Democratic victory in the 2020 US Presidential Election.</p><span class = "caption">Hashtags used to capture this phenomenon</span><p>#magacivilwar #marchfortrump #millionmagamarch #saveamerica #stopthesteal #stopthefraud</p></div>', unsafe_allow_html=True)


df_facebook_all, df_reddit_all, df_twitter_all, df_all, df_insurrection_before_all, df_insurrection_after_all, df_facebook_before_insurrection, df_facebook_after_insurrection, df_reddit_before_insurrection, df_reddit_after_insurrection, df_twitter_before_insurrection, df_twitter_after_insurrection = load_data()
df_facebook_before_election, df_facebook_after_election, df_reddit_before_election, df_reddit_after_election, df_twitter_before_election, df_twitter_after_election, df_facebook_all_election, df_reddit_all_election, df_twitter_all_election, df_all_election = load_data2()

# df_facebook_all = pd.concat([df_facebook_before_insurrection, df_facebook_after_insurrection])
# df_reddit_all = pd.concat([df_reddit_before_insurrection, df_reddit_after_insurrection])
# df_twitter_all = pd.concat([df_twitter_before_insurrection, df_twitter_after_insurrection])

event_selector=st.radio("Choose the topic you want to explore further",("Election Fraud", "Insurrection"))
if event_selector == 'Election Fraud':
    df_facebook_before = df_facebook_before_election
    df_facebook_after = df_facebook_after_election
    df_reddit_before = df_reddit_before_election
    df_reddit_after = df_reddit_after_election
    df_twitter_after = df_twitter_after_election
    df_twitter_before = df_twitter_before_election
else:
    df_facebook_before = df_facebook_before_insurrection
    df_facebook_after = df_facebook_after_insurrection
    df_reddit_before = df_reddit_before_insurrection
    df_reddit_after = df_reddit_after_insurrection
    df_twitter_after = df_twitter_after_insurrection
    df_twitter_before = df_twitter_before_insurrection

st.markdown("### Number of related posts in social medias we covered")


# Statistics of each social media platform
stat_col1, stat_col2, stat_col3 = st.columns(3)

# METRICS
facebook_total = len(df_facebook_before) + len(df_facebook_after)
reddit_total = len(df_reddit_before) + len(df_reddit_after)
twitter_total = len(df_twitter_before) + len(df_twitter_after)
stat_col1.metric(label = "Facebook", value=f"{facebook_total:,}")
stat_col2.metric(label = "Reddit", value=f"{reddit_total:,}")
stat_col3.metric(label = "Twitter", value=f"{twitter_total:,}")

st.write('<style> h4{padding: 0.2rem 0px 1rem} </style>', unsafe_allow_html=True)

stat_breakdown_col1, stat_breakdown_col2, stat_breakdown_col3, stat_breakdown_col4, stat_breakdown_col5, stat_breakdown_col6 = st.columns(6)
stat_breakdown_col1.markdown('Before')
stat_breakdown_col1.markdown('#### 'f"{len(df_facebook_before):,}")

stat_breakdown_col2.markdown('After')
stat_breakdown_col2.markdown('#### 'f"{len(df_facebook_after):,}")

stat_breakdown_col3.markdown('Before')
stat_breakdown_col3.markdown('#### 'f"{len(df_reddit_before):,}")

stat_breakdown_col4.markdown('After')
stat_breakdown_col4.markdown('#### 'f"{len(df_reddit_after):,}")

stat_breakdown_col5.markdown('Before')
stat_breakdown_col5.markdown('#### 'f"{len(df_twitter_before):,}")

stat_breakdown_col6.markdown('After')
stat_breakdown_col6.markdown('#### 'f"{len(df_twitter_after):,}")


# LINECHART
df_fb_count = pd.concat([df_facebook_before[['id','created_at']],df_facebook_after[['id','created_at']]])
df_rd_count = pd.concat([df_reddit_before[['id','created_at']],df_reddit_after[['id','created_at']]])
df_tw_count = pd.concat([df_twitter_before[['id','created_at']],df_twitter_after[['id','created_at']]])


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

st.markdown('In general, Twitter generated the most content and had an increasing number of posts over time around these topics, followed by a peak in content just before the Jan 6 insurrection. The number of posts between Facebook and Reddit throughout the election over time remained similar, but they peaked during the day of the insurrection (for Facebook), and the following day in Reddit. We see that discourse for election fraud started materializing only after the election itself, while talks of the insurrection started from way before and peaked on the D-day')

# SLIDER

min_date = min(df_linechart.index)
max_date = max(df_linechart.index)

st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)

st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{padding:0 20px;}</style>', unsafe_allow_html=True)

st.write('<style>div[role=radiogroup]{background-color:#eaeaea;border-radius:10px;padding:10px}</style>', unsafe_allow_html=True)



date_format = 'YYYY-MM-DD'


geo_after_df = pd.read_csv('./data/after_with_places_ext.csv')
geo_before_df = pd.read_csv('./data/before_with_places_ext.csv')

st.markdown("""---""")
time_container = st.container()
time_container.markdown('### Discourse over time')
time_container.markdown('In this section, we explore the top posting accounts and keywords over time. We also explore the regions in which geo-tagged discourse are concentrated on.')

choose=time_container.radio("Social media filter",("Facebook","Reddit","Twitter"), index=2)

date_filter = time_container.slider('Select date range', min_value=min_date, max_value=max_date, value=(min_date, max_date), format=date_format)

geo_all=(pd.concat([geo_before_df,geo_after_df]))
geo_all = geo_all[geo_all['country_code']=='US']

if choose == "Facebook":
    df_author = pd.concat([df_facebook_before[['id','author_id',DATE_COLUMN]],df_facebook_after[['id','author_id',DATE_COLUMN]]])

elif choose == "Reddit":
    df_author = pd.concat([df_reddit_before[['id','author_id',DATE_COLUMN]],df_reddit_after[['id','author_id',DATE_COLUMN]]])

else:
    df_author = pd.concat([df_twitter_before[['id','author_id',DATE_COLUMN]],df_twitter_after[['id','author_id',DATE_COLUMN]]])


df_author[DATE_COLUMN] = df_author[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
df_author[DATE_COLUMN] = pd.to_datetime(df_author[DATE_COLUMN])

date_mask = (df_author[DATE_COLUMN].dt.date >= date_filter[0]) & (df_author[DATE_COLUMN].dt.date <= date_filter[1])
df_author = df_author[date_mask]

top_authors = df_author.groupby(df_author['author_id']).count().drop(columns=DATE_COLUMN).reset_index().sort_values(by=['id'], ascending = False)


top_authors = top_authors.head()
color = ['#003f5c','#374c80','#7a5195','#bc5090','#ef5675']

top_authors['color'] = color

import altair as alt

author_bars = alt.Chart(top_authors).mark_bar().encode(
    x=alt.X('id', axis=alt.Axis(title='Number of posts')),
    y=alt.Y('author_id', axis=alt.Axis(title='Post Authors'), sort='-x'),
            color=alt.Color('color', scale=None)
)

text = author_bars.mark_text(
    align='left',
    baseline='middle',
    dx=3  # Nudges text to right so it doesn't appear on top of the bar
).encode(
    text='wheat:Q'
)

time_container.markdown('##### Top posting accounts')
time_container.altair_chart(author_bars, use_container_width = True)


if choose == "Facebook":
    time_container.markdown('Top posting accounts in Facebook contains many community pages in support of Trump.')
elif choose == "Reddit":
    time_container.markdown('Top posting accounts in Reddit consists of a lot of bots and some right-wing trolls. A good percentage of these accounts are currently suspended.')
else:
    time_container.markdown('Top posting accounts from Twitter are self-styled patriots from both the right wing and the left wing, and some sock puppets that have since deleted their posts.')

# Top keywords
#stopwords = set(STOPWORDS)
#stopwords.update(["https", "t", "co", "let", "will", "s", "use", "take", "used", "people", "said",
#            "say", "wasnt", "go", "well", "thing", "amp", "put", "&", "even", "Yet"])
#word_cleaning_before = ' '.join(text for text in df_all_before['text'])
#word_cleaning_before = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",word_cleaning_before)


if choose == "Facebook":
    df_content = pd.concat([df_facebook_before[['id','text',DATE_COLUMN]],df_facebook_after[['id','text',DATE_COLUMN]]])

elif choose == "Reddit":
    df_content= pd.concat([df_reddit_before[['id','text',DATE_COLUMN]],df_reddit_after[['id','text',DATE_COLUMN]]])
else:
    df_content = pd.concat([df_twitter_before[['id','text',DATE_COLUMN]],df_twitter_after[['id','text',DATE_COLUMN]]])

df_content[DATE_COLUMN] = df_content[DATE_COLUMN].apply(lambda x: x.replace('+00:00', ''))
df_content[DATE_COLUMN] = pd.to_datetime(df_content[DATE_COLUMN])

date_mask = (df_content[DATE_COLUMN].dt.date >= date_filter[0]) & (df_content[DATE_COLUMN].dt.date <= date_filter[1])
df_content = df_content[date_mask]

stopwords = set(STOPWORDS)
stopwords.update(["https", "t", "co", "let", "will", "s", "use", "take", "used", "people", "said",
            "say", "wasnt", "go", "well", "thing", "amp", "put", "&", "even", "Yet"])
word_cleaning = ' '.join(text for text in df_content['text'])
word_cleaning = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",word_cleaning)

wordcloud = WordCloud(stopwords=stopwords, max_words=10, width=800, height=400).generate(word_cleaning)
topwords_dict = wordcloud.words_
color_word = ['#0a010c','#bf974c','#223c84','#b6ba87','#a55340','#3d3726','#d1abb0','#484454', '#5a2d5e','#307026']


df_wordchart = {'term': list(topwords_dict.keys()), 'normalized_count': list(topwords_dict.values()), 'color': color_word}
df_wordchart = pd.DataFrame(data=df_wordchart)

text_bars = alt.Chart(df_wordchart).mark_bar().encode(
    x=alt.X('normalized_count', axis=alt.Axis(title='Normalized count')),
    y=alt.Y('term', axis=alt.Axis(title='Terms'), sort='-x'),
            color=alt.Color('color', scale=None)
)

text = text_bars.mark_text(
    align='left',
    baseline='middle',
    dx=3  # Nudges text to right so it doesn't appear on top of the bar
).encode(
    text='wheat:Q'
)

time_container.markdown('##### Top terms in posts')
time_container.altair_chart(text_bars, use_container_width = True)

#time_container.markdown('In this section we can see top posting accounts across time periods. Twitter ')

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

if choose == 'Twitter':
    flag = True
else:
    flag = False


if event_selector == 'Insurrection':
    df_twitter_before_ext = pd.read_csv('./data/after_with_places_ext.csv')
    df_twitter_after_ext = pd.read_csv('./data/before_with_places_ext.csv')

elif event_selector == 'Election Fraud':
    df_twitter_before_ext = pd.read_csv('./data/after_with_places_ext_election.csv')
    df_twitter_after_ext = pd.read_csv('./data/before_with_places_ext_election.csv')

if flag:
    df_tw_geo = pd.concat([df_twitter_before_ext[['id','created_at_x','place_type','country','name','full_name']],df_twitter_after_ext[['id','created_at_x','place_type','country','name','full_name']]])


    df_tw_geo[DATE_COLUMN] = df_tw_geo['created_at_x'].apply(lambda x: x.replace('+00:00', ''))
    df_tw_geo[DATE_COLUMN] = pd.to_datetime(df_tw_geo[DATE_COLUMN])
    date_mask = (df_tw_geo[DATE_COLUMN].dt.date >= date_filter[0]) & (df_tw_geo[DATE_COLUMN].dt.date <= date_filter[1])
    df_tw_geo = df_tw_geo[date_mask]

    df_tw_geo = df_tw_geo[df_tw_geo['country'] == 'United States']

    #process state column from cities
    df_tw_geo_citystate = df_tw_geo[df_tw_geo['place_type']=='city']
    df_tw_geo_adminstate = df_tw_geo[df_tw_geo['place_type']=='admin']

    df_tw_geo_citystate['state_abbrev'] = df_tw_geo_citystate['full_name'].str.split(',').str[1].str.strip()
    df_tw_geo_adminstate['state_name'] = df_tw_geo_adminstate['name'].copy()
    df_tw_geo_citystate['state_name'] = df_tw_geo_citystate['state_abbrev'].map(inverted_us_state)
    df_tw_geo_adminstate['state_abbrev'] = df_tw_geo_adminstate['state_name'].map(us_state_to_abbrev)
    df_tw_geo_citystate['state_name'] = df_tw_geo_citystate['state_abbrev'].map(inverted_us_state)

    df_tw_geo = pd.concat([df_tw_geo_adminstate,df_tw_geo_citystate])

    #st.write(df_tw_geo)

    top_states = df_tw_geo.groupby(df_tw_geo['state_abbrev']).count().drop(columns=[DATE_COLUMN,'place_type','name','full_name','created_at_x','state_name','country']).reset_index().sort_values(by=['id'], ascending = False)

    #st.write(top_states)


    fig = go.Figure(data=go.Choropleth(
        locations=top_states['state_abbrev'],
        z=top_states['id'].astype(int),
        locationmode='USA-states',
        colorscale='Reds',
        autocolorscale=False, # hover text
        marker_line_color='white', # line markers between states
        colorbar_title="# of Tweets"
    ))

    fig.update_layout(
        title_text='Discourse in US States',
        geo = dict(
            scope='usa',
            projection=go.layout.geo.Projection(type = 'albers usa'),
            showlakes=True, # lakes
            lakecolor='rgb(255, 255, 255)'),
    )
    time_container.markdown('##### Geographic concentration of posts')

    time_container.plotly_chart(fig)

    time_container.markdown('With the map, we visualize the concentration of geo-tagged posts. During the events studied, the geotagged posts mainly come from the states of California, Florida, Texas and New York. This aligned with the general population numbers - those are top 4 states in terms of population. An interesting thing to analyze here is that people in Florida seems to be more active in tweeting in this discourse than its supposed proportion in population - they have less people than Texas, yet they tweet more.')

else:
    time_container.markdown('##### Geographic concentration of posts')
    time_container.warning('Geotagged data are only available in the Twitter dataset')

st.markdown("""---""")

emotion_container = st.container()
emotion_container.markdown('### Emotions in discourse')
emotion_container.markdown('We next explore emotions that are present in the discourse over time: the share of positive vs negative sentiments; and the dominant emotion (sadness, joy, disgust, anger and fear) in each post.')
social_selector=emotion_container.radio("Social media filter",("Facebook","Reddit","Twitter"), index=2, key = "socialselectemo")
emotion_container.markdown('##### Sentiments over time')

box_posneg_1, box_emotion_up_space, box_posneg_2 = emotion_container.columns((2, 0.1, 1))
values = ['negative', 'positive', 'neutral']

if event_selector == 'Insurrection':
    with box_posneg_1:
        if social_selector == 'Facebook':
            conditions = [
            (df_facebook_all['liwc.posemo'] < df_facebook_all['liwc.negemo']),
            (df_facebook_all['liwc.posemo'] > df_facebook_all['liwc.negemo']),
            (df_facebook_all['liwc.posemo'] == df_facebook_all['liwc.negemo'])
            ]
            sentiment_table = df_facebook_all.copy()
            sentiment_table['sentiment'] = np.select(conditions, values)
            sentiment_table['pos'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
            sentiment_table['neg'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='negative' else 0)
            sentiment_table['neutral'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='neutral' else 0)
            df_stack = pd.DataFrame({'pos': sentiment_table['pos'].tolist(),'neg': sentiment_table['neg'].tolist(),'neutral': sentiment_table['neutral'].tolist(), 'date':sentiment_table['date_column'].tolist() })
            df_stack = df_stack.groupby('date').agg('sum')
            box_posneg_1.bar_chart(df_stack)
            df_stack.plot.bar(stacked=True)
        elif social_selector == 'Reddit':
            conditions = [
            (df_reddit_all['liwc.posemo'] < df_reddit_all['liwc.negemo']),
            (df_reddit_all['liwc.posemo'] > df_reddit_all['liwc.negemo']),
            (df_reddit_all['liwc.posemo'] == df_reddit_all['liwc.negemo'])
            ]
            sentiment_table = df_reddit_all.copy()
            sentiment_table['sentiment'] = np.select(conditions, values)
            sentiment_table['pos'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
            sentiment_table['neg'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='negative' else 0)
            sentiment_table['neutral'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='neutral' else 0)
            df_stack = pd.DataFrame({'pos': sentiment_table['pos'].tolist(),'neg': sentiment_table['neg'].tolist(),'neutral': sentiment_table['neutral'].tolist(), 'date':sentiment_table['date_column'].tolist() })
            df_stack = df_stack.groupby('date').agg('sum')
            box_posneg_1.bar_chart(df_stack)
            df_stack.plot.bar(stacked=True)
        elif social_selector == 'Twitter':
            conditions = [
            (df_twitter_all['liwc.posemo'] < df_twitter_all['liwc.negemo']),
            (df_twitter_all['liwc.posemo'] > df_twitter_all['liwc.negemo']),
            (df_twitter_all['liwc.posemo'] == df_twitter_all['liwc.negemo'])
            ]
            sentiment_table = df_twitter_all.copy()
            sentiment_table['sentiment'] = np.select(conditions, values)
            sentiment_table['pos'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
            sentiment_table['neg'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='negative' else 0)
            sentiment_table['neutral'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='neutral' else 0)
            df_stack = pd.DataFrame({'pos': sentiment_table['pos'].tolist(),'neg': sentiment_table['neg'].tolist(),'neutral': sentiment_table['neutral'].tolist(), 'date':sentiment_table['date_column'].tolist() })
            df_stack = df_stack.groupby('date').agg('sum')
            box_posneg_1.bar_chart(df_stack)
            df_stack.plot.bar(stacked=True)
    with box_posneg_2:
        if social_selector == 'Facebook':
            box_posneg_2.markdown('In the discourse around insurrection, the sentiments kept are increasingly positive from Dec. 31, 2020 to Jan. 5, 2021, the day before insurrection. The total messages related to insurrection increased 240%, which led to highest peak of sentiments on the day before insurrection. The positive sentiment decrease from 57.8% to 51.9% and the negative sentiment increase from 29.7% to 32.5%.')
        elif social_selector == 'Reddit':
            box_posneg_2.markdown('In the discourse around insurrection, the sentiments kept are increasingly positive from Dec. 31, 2020 to Jan. 6, 2021, the day of  insurrection. The total messages related to insurrection increased 550%, which led to highest peak of sentiments on the day of insurrection. The positive sentiment decrease from 35% to 33% and the negative sentiment increase from 33.3% to 52%.')
        elif social_selector == 'Twitter':
            box_posneg_2.markdown('The total messages related to insurrection increased 303%, which led to highest peak of sentiments on the day of insurrection. The positive sentiment decrease from 40.7% to 32% and the negative sentiment increase from 44.07% to 51.2%.')

elif event_selector == 'Election Fraud':
    with box_posneg_1:
        if social_selector == 'Facebook':

            conditions = [
            (df_facebook_all_election['liwc.posemo'] < df_facebook_all_election['liwc.negemo']),
            (df_facebook_all_election['liwc.posemo'] > df_facebook_all_election['liwc.negemo']),
            (df_facebook_all_election['liwc.posemo'] == df_facebook_all_election['liwc.negemo'])
            ]
            sentiment_table = df_facebook_all_election.copy()
            sentiment_table['sentiment'] = np.select(conditions, values)
            sentiment_table['pos'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
            sentiment_table['neg'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='negative' else 0)
            sentiment_table['neutral'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='neutral' else 0)
            df_stack = pd.DataFrame({'pos': sentiment_table['pos'].tolist(),'neg': sentiment_table['neg'].tolist(),'neutral': sentiment_table['neutral'].tolist(), 'date':sentiment_table['date_column'].tolist() })
            df_stack = df_stack.groupby('date').agg('sum')
            box_posneg_1.bar_chart(df_stack)
            df_stack.plot.bar(stacked=True)
        elif social_selector == 'Reddit':

            conditions = [
            (df_reddit_all_election['liwc.posemo'] < df_reddit_all_election['liwc.negemo']),
            (df_reddit_all_election['liwc.posemo'] > df_reddit_all_election['liwc.negemo']),
            (df_reddit_all_election['liwc.posemo'] == df_reddit_all_election['liwc.negemo'])
            ]
            sentiment_table = df_reddit_all_election.copy()
            sentiment_table['sentiment'] = np.select(conditions, values)
            sentiment_table['pos'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
            sentiment_table['neg'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='negative' else 0)
            sentiment_table['neutral'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='neutral' else 0)
            df_stack = pd.DataFrame({'pos': sentiment_table['pos'].tolist(),'neg': sentiment_table['neg'].tolist(),'neutral': sentiment_table['neutral'].tolist(), 'date':sentiment_table['date_column'].tolist() })
            df_stack = df_stack.groupby('date').agg('sum')
            box_posneg_1.bar_chart(df_stack)
            df_stack.plot.bar(stacked=True)
        elif social_selector == 'Twitter':

            conditions = [
            (df_twitter_all_election['liwc.posemo'] < df_twitter_all_election['liwc.negemo']),
            (df_twitter_all_election['liwc.posemo'] > df_twitter_all_election['liwc.negemo']),
            (df_twitter_all_election['liwc.posemo'] == df_twitter_all_election['liwc.negemo'])
            ]
            sentiment_table = df_twitter_all_election.copy()
            sentiment_table['sentiment'] = np.select(conditions, values)
            sentiment_table['pos'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='positive' else 0)
            sentiment_table['neg'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='negative' else 0)
            sentiment_table['neutral'] = sentiment_table['sentiment'].apply(lambda x: 1 if x=='neutral' else 0)
            df_stack = pd.DataFrame({'pos': sentiment_table['pos'].tolist(),'neg': sentiment_table['neg'].tolist(),'neutral': sentiment_table['neutral'].tolist(), 'date':sentiment_table['date_column'].tolist() })
            df_stack = df_stack.groupby('date').agg('sum')
            box_posneg_1.bar_chart(df_stack)
            df_stack.plot.bar(stacked=True)
    with box_posneg_2:
        if social_selector == 'Facebook':
            box_posneg_2.markdown('After the election, the sentiments of the discourse related to election skyrocketed. The sentiment increased 1744% from Oct. 31, 2020 to Nov. 5, 2020. The highest peak is on Nov 5, 2020.')
        elif social_selector == 'Reddit':
            box_posneg_2.markdown('After the election, the sentiments of the discourse related to election skyrocketed. The sentiment 942% from Oct. 31, 2020 to Nov. 5, 2020. The highest peak is on Nov 5, 2020.')
        elif social_selector == 'Twitter':
            box_posneg_2.markdown('After the election, the sentiments of the discourse related to election skyrocketed. The sentiment increased 1982% from Oct. 31, 2020 to Nov. 5, 2020. The highest peak is on Nov 5, 2020.')



emotion_container.markdown('##### Share of emotions over time')

box_emotion_up_1, box_emotion_up_space, box_emotion_up_2 = emotion_container.columns((2, 0.1, 1))
if event_selector == 'Insurrection':
    with box_emotion_up_1:
        if social_selector == 'Facebook':
            df_emo = pd.DataFrame(df_facebook_all['highest_emotion'].value_counts()).reset_index()
            df_emo.columns = ['emotions', 'frequency']
            df_emo['color'] = ['#c489be','#4317aa','#452a49','#686860','#54a24b']


            emo_bars = alt.Chart(df_emo).mark_bar().encode(
                x=alt.X('frequency', axis=alt.Axis(title='Frequency')),
                y=alt.Y('emotions', axis=alt.Axis(title='Terms'), sort='-x'),
                        color=alt.Color('color', scale=None)
            )


            box_emotion_up_1.altair_chart(emo_bars, use_container_width = True)

        elif social_selector == 'Reddit':
            df_emo = pd.DataFrame(df_reddit_all['highest_emotion'].value_counts()).reset_index()
            df_emo.columns = ['emotions', 'frequency']
            df_emo['color'] = ['#c489be','#4317aa','#452a49','#686860','#54a24b']


            emo_bars = alt.Chart(df_emo).mark_bar().encode(
                x=alt.X('frequency', axis=alt.Axis(title='Frequency')),
                y=alt.Y('emotions', axis=alt.Axis(title='Terms'), sort='-x'),
                        color=alt.Color('color', scale=None)
            )


            box_emotion_up_1.altair_chart(emo_bars, use_container_width = True)
        elif social_selector == 'Twitter':
            df_emo = pd.DataFrame(df_twitter_all['highest_emotion'].value_counts()).reset_index()
            df_emo.columns = ['emotions', 'frequency']
            df_emo['color'] = ['#c489be','#4317aa','#452a49','#686860','#54a24b']


            emo_bars = alt.Chart(df_emo).mark_bar().encode(
                x=alt.X('frequency', axis=alt.Axis(title='Frequency')),
                y=alt.Y('emotions', axis=alt.Axis(title='Terms'), sort='-x'),
                        color=alt.Color('color', scale=None)
            )


            box_emotion_up_1.altair_chart(emo_bars, use_container_width = True)


    with box_emotion_up_2:
        box_emotion_up_2.markdown('The dominating emotion on social media discourse is disgust, leading to an inference that people turn to social media to voice their discontent.')
    if social_selector == 'Facebook':
        df_stack = pd.DataFrame({'sadness': df_facebook_all['emotion.sadness'].tolist(),'anger': df_facebook_all['emotion.anger'].tolist(),'disgust': df_facebook_all['emotion.disgust'].tolist(),'joy': df_facebook_all['emotion.joy'].tolist(),'fear': df_facebook_all['emotion.fear'].tolist(), 'date':df_facebook_all['date_column'].tolist() })
        df_stack = df_stack.groupby('date').agg('sum')
            # fig = px.bar(df_stack, x="date", y=["sadness", "anger", "disgust"])
            # st.write(fig)
        emotion_container.bar_chart(df_stack)
    elif social_selector == 'Reddit':
        df_stack = pd.DataFrame({'sadness': df_reddit_all['emotion.sadness'].tolist(),'anger': df_reddit_all['emotion.anger'].tolist(),'disgust': df_reddit_all['emotion.disgust'].tolist(),'joy': df_reddit_all['emotion.joy'].tolist(),'fear': df_reddit_all['emotion.fear'].tolist(), 'date':df_reddit_all['date_column'].tolist() })
        df_stack = df_stack.groupby('date').agg('sum')
        emotion_container.bar_chart(df_stack)
    elif social_selector == 'Twitter':
        df_stack = pd.DataFrame({'sadness': df_twitter_all['emotion.sadness'].tolist(),'anger': df_twitter_all['emotion.anger'].tolist(),'disgust': df_twitter_all['emotion.disgust'].tolist(),'joy': df_twitter_all['emotion.joy'].tolist(),'fear': df_twitter_all['emotion.fear'].tolist(), 'date':df_twitter_all['date_column'].tolist() })
        df_stack = df_stack.groupby('date').agg('sum')
        emotion_container.bar_chart(df_stack)

elif event_selector == 'Election Fraud':
    with box_emotion_up_1:
        if social_selector == 'Facebook':
            df_emo = pd.DataFrame(df_facebook_all_election['highest_emotion'].value_counts()).reset_index()
            df_emo.columns = ['emotions', 'frequency']
            df_emo['color'] = ['#c489be','#4317aa','#452a49','#686860','#54a24b']


            emo_bars = alt.Chart(df_emo).mark_bar().encode(
                x=alt.X('frequency', axis=alt.Axis(title='Frequency')),
                y=alt.Y('emotions', axis=alt.Axis(title='Terms'), sort='-x'),
                        color=alt.Color('color', scale=None)
            )


            box_emotion_up_1.altair_chart(emo_bars, use_container_width = True)
        elif social_selector == 'Reddit':
            df_emo = pd.DataFrame(df_reddit_all_election['highest_emotion'].value_counts()).reset_index()
            df_emo.columns = ['emotions', 'frequency']
            df_emo['color'] = ['#c489be','#4317aa','#452a49','#686860','#54a24b']


            emo_bars = alt.Chart(df_emo).mark_bar().encode(
                x=alt.X('frequency', axis=alt.Axis(title='Frequency')),
                y=alt.Y('emotions', axis=alt.Axis(title='Terms'), sort='-x'),
                        color=alt.Color('color', scale=None)
            )


            box_emotion_up_1.altair_chart(emo_bars, use_container_width = True)
        elif social_selector == 'Twitter':
            df_emo = pd.DataFrame(df_reddit_all_election['highest_emotion'].value_counts()).reset_index()
            df_emo.columns = ['emotions', 'frequency']
            df_emo['color'] = ['#c489be','#4317aa','#452a49','#686860','#54a24b']


            emo_bars = alt.Chart(df_emo).mark_bar().encode(
                x=alt.X('frequency', axis=alt.Axis(title='Frequency')),
                y=alt.Y('emotions', axis=alt.Axis(title='Terms'), sort='-x'),
                        color=alt.Color('color', scale=None)
            )


            box_emotion_up_1.altair_chart(emo_bars, use_container_width = True)

    with box_emotion_up_2:
        box_emotion_up_2.markdown('The dominating emotion on social media discourse is disgust, leading to an inference that people turn to social media to voice their discontent.')
    if social_selector == 'Facebook':
        df_stack = pd.DataFrame({'sadness': df_facebook_all_election['emotion.sadness'].tolist(),'anger': df_facebook_all_election['emotion.anger'].tolist(),'disgust': df_facebook_all_election['emotion.disgust'].tolist(),'joy': df_facebook_all_election['emotion.joy'].tolist(),'fear': df_facebook_all_election['emotion.fear'].tolist(), 'date':df_facebook_all_election['date_column'].tolist() })
        df_stack = df_stack.groupby('date').agg('sum')
        # fig = px.bar(df_stack, x="date", y=["sadness", "anger", "disgust"])
        # st.write(fig)
        emotion_container.bar_chart(df_stack)
    elif social_selector == 'Reddit':
        df_stack = pd.DataFrame({'sadness': df_reddit_all_election['emotion.sadness'].tolist(),'anger': df_reddit_all_election['emotion.anger'].tolist(),'disgust': df_reddit_all_election['emotion.disgust'].tolist(),'joy': df_reddit_all_election['emotion.joy'].tolist(),'fear': df_reddit_all_election['emotion.fear'].tolist(), 'date':df_reddit_all_election['date_column'].tolist() })
        df_stack = df_stack.groupby('date').agg('sum')
        emotion_container.bar_chart(df_stack)
    elif social_selector == 'Twitter':
        df_stack = pd.DataFrame({'sadness': df_twitter_all_election['emotion.sadness'].tolist(),'anger': df_twitter_all_election['emotion.anger'].tolist(),'disgust': df_twitter_all_election['emotion.disgust'].tolist(),'joy': df_twitter_all_election['emotion.joy'].tolist(),'fear': df_twitter_all_election['emotion.fear'].tolist(), 'date':df_twitter_all_election['date_column'].tolist() })
        df_stack = df_stack.groupby('date').agg('sum')
        emotion_container.bar_chart(df_stack)



emotion_container.markdown('##### Intensity of emotions over time')
emotion_container.markdown('In this section we explore the intensity of the emotions over the course of the discourse in the selected event.')

emotion_selector = emotion_container.radio("Pick an emotion you want to focus on", ('Sadness', 'Anger', 'Disgust', 'Fear', 'Joy'))

box_emotion_1, box_emotion_space, box_emotion_2 = emotion_container.columns((2, 0.1, 1))

df_insurrection_before_all_len = len(df_insurrection_before_all)
df_insurrection_after_all_len = len(df_insurrection_after_all)

with box_emotion_1:
    box_emotion_1.markdown('_Box plot showing intensity of ' + str.lower(emotion_selector) + ' in '+ social_selector + ' post around '+ str.lower(event_selector) + ' over time_')
    if social_selector == 'Facebook': social_selector_naming='fb'
    elif social_selector == 'Reddit': social_selector_naming='rd'
    elif social_selector == 'Twitter': social_selector_naming='tw'
    box_emotion_1.image('images/{}_{}_{}.png'.format(social_selector_naming, event_selector.lower().replace(' fraud', ''), emotion_selector.lower()))

with box_emotion_2:
    box_emotion_2.markdown('<p style=font-weight:bold;font-size:1.5rem;>{}<p>'.format(emotion_selector.upper()), unsafe_allow_html=True)
    if event_selector == 'Insurrection':
        if social_selector == 'Facebook':
            box_emotion_2.markdown('Disgust and anger increases after the insurrection. Sadness, joy and fear peaked on the day of insurrection.')
        elif social_selector == 'Reddit':
            box_emotion_2.markdown('Disgust and Anger increases after the insurrection; Joy and Fear decreases after the insurrection; the intensity of Sadness remains constant.')
        elif social_selector == 'Twitter':
            box_emotion_2.markdown('Disgust and Anger intensifies after the insurrection; Joy and Fear decreases after the insurrection; the intensity of Sadness remains constant.')
    elif event_selector == 'Election Fraud':
        if social_selector == 'Facebook':
            box_emotion_2.markdown('Disgust and Anger intensifies after the election; Joy peaked on the day of the elections; Sadness and Anger decreases in intensity after the election.')
        elif social_selector == 'Reddit':
            box_emotion_2.markdown('Disgust and Joy intensifies after the election; Anger peaked the day before the election but plunged on the day of the election; Fear decreases after the election while the intensity of sadness remained the same.')
        elif social_selector == 'Twitter':
            box_emotion_2.markdown('Disgust and Anger increases after the election; Sadness, Joy and Fear decreased in intensity after the election.')


emotion_container.markdown('##### Sample post from {} that contains an emotion of {}'.format(social_selector, emotion_selector.upper()))
emotion_container.write("<style>.post-sample{background: #eaeaea; padding: 10px; border: 1px solid #eaeaea; border-radius: 10px; margin-bottom: 10px} </style>", unsafe_allow_html=True)
sample_1, sample_space, sample_2 = emotion_container.columns((1, 0.1, 1))

if social_selector == "Facebook":
    logo = "https://img.icons8.com/color/48/000000/facebook-new.png"
    post_before = df_facebook_before[df_facebook_before['highest_emotion']==emotion_selector.lower()].sample()
    post_after = df_facebook_after[df_facebook_after['highest_emotion']==emotion_selector.lower()].sample()

elif social_selector == "Twitter":
    logo = "https://maxcdn.icons8.com/Color/PNG/48/Social_Networks/twitter-48.png"
    post_before = df_twitter_before[df_twitter_before['highest_emotion']==emotion_selector.lower()].sample()
    post_after = df_twitter_after[df_twitter_after['highest_emotion']==emotion_selector.lower()].sample()

elif social_selector == "Reddit":
    logo = "https://img.icons8.com/doodle/48/000000/reddit--v4.png"
    post_before = df_reddit_before[df_reddit_before['highest_emotion']==emotion_selector.lower()].sample()
    post_after = df_facebook_after[df_twitter_after['highest_emotion']==emotion_selector.lower()].sample()


with sample_1:
    sample_1.markdown('<p style=font-weight:bold;font-size:1rem;color:grey;> Sample post with emotion of {} before the {} <p>'.format(emotion_selector.upper(), event_selector.lower().replace(' fraud', '')), unsafe_allow_html=True)
    post_before_text = post_before['text'].values[0]
    post_before_user = post_before['author_id'].values[0]
    sample_1.image(logo)
    sample_1.markdown('**'+post_before_user + '**:')
    sample_1.write("<div class = '{}-post post-sample'><p>"+post_before_text+"</p></div>".format(social_selector), unsafe_allow_html=True)
with sample_2:
    sample_2.markdown('<p style=font-weight:bold;font-size:1rem;color:grey;> Sample post with emotion of {} after the {} <p>'.format(emotion_selector.upper(), event_selector.lower().replace(' fraud', '')), unsafe_allow_html=True)
    post_after_text = post_after['text'].values[0]
    post_after_user = post_after['author_id'].values[0]
    sample_2.image(logo)
    sample_2.markdown('**'+post_after_user + '**:')
    sample_2.write("<div class = '{}-post post-sample'><p>"+post_after_text+"</p></div>".format(social_selector), unsafe_allow_html=True)
