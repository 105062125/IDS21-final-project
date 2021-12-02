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

    df_facebook_before_insurrection = pd.read_csv('./data/facebook_before_insurrection.csv')
    df_facebook_after_insurrection = pd.read_csv('./data/facebook_after_insurrection.csv')
    df_reddit_before_insurrection = pd.read_csv('./data/reddit_before_insurrection.csv')
    df_reddit_after_insurrection = pd.read_csv('./data/reddit_after_insurrection.csv')
    df_twitter_before_insurrection = pd.read_csv('./data/twitter_before_insurrection.csv')
    df_twitter_after_insurrection = pd.read_csv('./data/twitter_after_insurrection.csv')

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

st.markdown('Talk about numbers, then which source generated the most content bla bla bla, Facebook peaks where, and desc of trend Twitter peaks where and desc of trend Reddit peaks where')

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
time_container.markdown('In this section we can see deeper on what happened with the discourse over time: seeing top posting accounts and keywords over the period of time, and for geo-tagged Twitter data, we can see where the discourse are concentrated on')

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
    time_container.markdown('In this section we can see top posting accounts across time periods. Facebook a lot of Trump-supporting community page ')
elif choose == "Reddit":
    time_container.markdown('In this section we can see top posting accounts across time periods. Reddit a lot of bots, some right-wing trolls that are suspended')
else:
    time_container.markdown('In this section we can see top posting accounts across time periods. Twitter suspicious, sock puppets')

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

time_container.markdown('In this section we can see top posting accounts across time periods. Twitter ')




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
emotion_container.markdown('### Emotions in discourse')
st.markdown('<p style=color:grey;font-size:1em;>Sentiment<p>', unsafe_allow_html=True)
box_posneg_1, box_emotion_up_space, box_posneg_2 = st.columns((2, 0.1, 1))
values = ['negative', 'positive', 'neutral']

if event_selector == 'Insurrection':
    with box_posneg_1:
        if choose == 'Facebook':

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
            st.bar_chart(df_stack)
            st.write(df_stack)
            df_stack.plot.bar(stacked=True)
        elif choose == 'Reddit':

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
            st.bar_chart(df_stack)
            df_stack.plot.bar(stacked=True)
        elif choose == 'Twitter':

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
            st.bar_chart(df_stack)
            df_stack.plot.bar(stacked=True)
    with box_posneg_2:
        st.markdown('TO DO: Explain what\'s going on in the boxplot Insurrection')

elif event_selector == 'Election':
    with box_posneg_1:
        if choose == 'Facebook':

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
            st.bar_chart(df_stack)
            df_stack.plot.bar(stacked=True)
        elif choose == 'Reddit':

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
            st.bar_chart(df_stack)
            df_stack.plot.bar(stacked=True)
        elif choose == 'Twitter':

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
            st.bar_chart(df_stack)
            df_stack.plot.bar(stacked=True)
    with box_posneg_2:
        st.markdown('TO DO: Explain what\'s going on in the boxplot Election')



st.markdown('<p style=color:grey;font-size:1em;>Emotion over time<p>', unsafe_allow_html=True)

box_emotion_up_1, box_emotion_up_space, box_emotion_up_2, box_emotion_up_3 = st.columns((2, 0.1, 2, 1))

if event_selector == 'Insurrection':
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

elif event_selector == 'Election':
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
