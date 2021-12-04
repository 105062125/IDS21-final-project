import numpy as np
import pandas as pd
import pickle
import bz2

df_facebook_before_insurrection = pd.read_csv('./data/facebook_before_insurrection.csv')
df_facebook_after_insurrection = pd.read_csv('./data/facebook_after_insurrection.csv')
df_reddit_before_insurrection = pd.read_csv('./data/reddit_before_insurrection.csv')
df_reddit_after_insurrection = pd.read_csv('./data/reddit_after_insurrection.csv')
df_twitter_before_insurrection = pd.read_csv('./data/twitter_before_insurrection.csv')
df_twitter_after_insurrection = pd.read_csv('./data/twitter_after_insurrection.csv')

with open('facebook_before_insurrection.pkl', 'wb') as f:
    compressed_file = bz2.BZ2File(f, 'w')
    pickle.dump(df_facebook_before_insurrection, f)

with open('facebook_after_insurrection.pkl', 'wb') as f:
    pickle.dump(df_facebook_after_insurrection, f)

with open('reddit_before_insurrection.pkl', 'wb') as f:
        pickle.dump(df_reddit_before_insurrection, f)

with open('reddit_after_insurrection.pkl', 'wb') as f:
        pickle.dump(df_reddit_after_insurrection, f)

with open('twitter_before_insurrection.pkl', 'wb') as f:
        pickle.dump(df_twitter_before_insurrection, f)

with open('twitter_after_insurrection.pkl', 'wb') as f:
        pickle.dump(df_twitter_after_insurrection, f)

df_facebook_before_election = pd.read_csv('./data/facebook_before_election.csv')
df_facebook_after_election = pd.read_csv('./data/facebook_after_election.csv')
df_reddit_before_election = pd.read_csv('./data/reddit_before_election.csv')
df_reddit_after_election = pd.read_csv('./data/reddit_after_election.csv')
df_twitter_before_election = pd.read_csv('./data/twitter_before_election.csv')
df_twitter_after_election = pd.read_csv('./data/twitter_after_election.csv')

with open('facebook_before_election.pkl', 'wb') as f:
    pickle.dump(df_facebook_before_election, f)

with open('facebook_after_election.pkl', 'wb') as f:
    pickle.dump(df_facebook_after_election, f)

with open('reddit_before_election.pkl', 'wb') as f:
        pickle.dump(df_reddit_before_election, f)

with open('reddit_after_election.pkl', 'wb') as f:
        pickle.dump(df_reddit_after_election, f)

with open('twitter_before_election.pkl', 'wb') as f:
        pickle.dump(df_twitter_before_election, f)

with open('twitter_after_election.pkl', 'wb') as f:
        pickle.dump(df_twitter_after_election, f)

df_twitter_before_ext = pd.read_csv('./data/before_with_places_ext.csv')
df_twitter_after_ext = pd.read_csv('./data/after_with_places_ext.csv')

df_twitter_before_ext_el = pd.read_csv('./data/before_with_places_ext_election.csv')
df_twitter_after_ext_el = pd.read_csv('./data/after_with_places_ext_election.csv')

with open('before_with_places_ext.pkl', 'wb') as f:
        pickle.dump(df_twitter_before_ext, f)

with open('after_with_places_ext.pkl', 'wb') as f:
        pickle.dump(df_twitter_after_ext, f)

with open('before_with_places_ext_election.pkl', 'wb') as f:
        pickle.dump(df_twitter_before_ext_el, f)

with open('after_with_places_ext_election.pkl', 'wb') as f:
        pickle.dump(df_twitter_after_ext_el, f)
