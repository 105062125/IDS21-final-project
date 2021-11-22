# The Emotions of the US Election and Insurrection
By Adya Danaditya, Lynnette Ng and Kevin Chen

andrew ids: adanadit, huixiann, weichieh

## Introduction
The 2020 US Elections was held on November 2020. We visualize the discourse on social media before and after the elections, related to the theme of voter fraud.
After the elections, on 6th January 2021, there was an insurrection at the Capitol in Washington DC. 
Protesters claimed voter fraud and that the wrong president was voted in. We visualize the discourse on social media before and after this insurrection event. 
Specifically, we looked at the changes in emotions, sentiment and other psycholinguistic cues during these few weeks. 
We also looked at the geographical distribution of these items across states in the United States.
We visualize three social media platforms: Twitter, Reddit and Facebook.

### Goals of the project
This data visualization aims to explore the following:
1. What is the general emotions and sentiment displayed in the social media data regarding voter fraud before/after the election? 
2. What is the general emotions and sentiment displayed in the social media data regarding voter fraud before/after the insurrection?
3. How do the general emotions and sentiment displayed in the social media data differ across states? 
4. How do the general emotions and sentiment displayed in the social media data differ across social media platforms? 

# Data
## Collection Methodology
In this project we explore several datasets spanning three social media and two events. These data are collected along the theme of voter fraud. 
The timeframe for election collection was: (a) before election: October 27 to November 2 2020; (b) after election: November 3 to November 9 2020. 
The time frame for insurrection collection was: (a) before insurrection: Dec 30 to Jan 5; (b) after insurrection: Jan 6 to Jan 12.

Twitter data was collected using the Twitter V1 API. Facebook data was collected with CrowdTangle. Reddit data was collected with PSAW.

The list of hashtags used in the collection includes: #corruptelection, #deadvoters, #deceasedvoters, #dominionvotingsystems, #electionfraud, #electionintegrity, #fakeelection, #fakevotes, #fraudulentelection, #legalvotesonly, #legitimatevotesonly, #massivecorruption, #riggedelection, #stolenelection, #trumpismypresident, #voterfraud.

After this, we also further rehydrate the Twitter data using the Python Twarc library to obtain the full tweets with geolocation information.

The data can be accessed from [Google Drive](https://drive.google.com/drive/folders/1FPgvlw2DOEcz3gKHHd0EL0hLV8cmkaCt?usp=sharing)

## Applying Machine Learning 
We look to apply machine learning to the dataset by further annotating the data with emotions.
We first train an emotion classifier on a tweet dataset that was already annotated. This is the [SemEval-2018 Task 1: Affect in Tweets](https://competitions.codalab.org/competitions/17751) dataset. We create a classifier using a BiLSTM and a Convolutional Neural Network. Our classifier performs with 74.8% accuracy. Then we apply this classifier to annotate each of the text with a probability of each of the following emotions: disgust, anger, fear, joy and sadness.

We further annotate the data with sentiments and psycholinguistic cues, which was done using the Linguistic Inquiry and Word Count engine. 
This engine returns normalized probabilities of each item, eg. positive sentiment, negative sentiment, expression of family. 

## Data Preprocessing 
After receiving the data, we performed data processing to clean the data in order to be able to better visualize it as well as for the practical reason of making the interactive process run in a quick manner. 
We removed all items with missing emotions and sentiments from the data. 
We also removed items with the same repeated texts, eg retweets. 
We created new columns tagging the prevailing sentiment and emotion of each text by processing sentiment and emotion measures of each text in the dataset and take the emotion or sentiment which is highest in value as the tag.

# Approach 
We will attempt to tell a compelling story about the differences in emotions across the two events, providing a side by side comparison. 
To give this contrast, we will divide the page into two for the user to discern the change of emotions before and after the events.
We will present data summaries and general emotion and sentiment summary by histogram and multiple metric views.
To present data regarding the states, we will use a map to show geography. We will overlay translucent bubbles of different colours to represent the different emotions.

# Exploratory Data Analysis
Before showing the interactive visualization, we did a round of exploratory data analysis to get a sense of the data. The EDA for the election part of the data can be seen [here](Election_EDA.ipynb), meanwhile the EDA for the insurrection part of the data can be seen [here](insurrection_exploration.ipynb)

