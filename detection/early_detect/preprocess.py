import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tweepy
import networkx as nx
nltk.download('stopwords')
nltk.download('wordnet')

# Define preprocessing function
def preprocess_text(text):
    """
    Preprocess text by removing URLs, mentions, special characters, stop words, and lemmatizing words.

    Parameters:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    # Remove URLs and mentions
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-z ]+', '', text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    return text



def create_full_network(api_key, api_secret_key, access_token, access_token_secret, tweet_ids):
    """
    Hydrates tweets and creates the full network of retweets, mentions, and follows.

    Parameters:
        api_key (str): Your Twitter API key.
        api_secret_key (str): Your Twitter API secret key.
        access_token (str): Your Twitter access token.
        access_token_secret (str): Your Twitter access token secret.
        tweet_ids (list): A list of tweet IDs to hydrate.

    Returns:
        networkx.DiGraph: The full network of retweets, mentions, and follows.
    """
    # Authenticate with Twitter API
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # Hydrate tweets
    tweets = api.statuses_lookup(tweet_ids, tweet_mode='extended')

    # Create graph
    G = nx.DiGraph()

    # Add nodes and edges for retweets, mentions, and follows
    for tweet in tweets:
        G.add_node(tweet.id_str)
        user_id = tweet.user.id_str
        G.add_node(user_id)
        G.add_edge(user_id, tweet.id_str, type='tweeted')
        for mention in tweet.entities['user_mentions']:
            mention_id = mention['id_str']
            G.add_node(mention_id)
            G.add_edge(user_id, mention_id, type='mentioned')
        retweets = api.retweets(tweet.id, tweet_mode='extended')
        for retweet in retweets:
            retweet_user_id = retweet.user.id_str
            G.add_node(retweet_user_id)
            G.add_edge(retweet_user_id, tweet.id_str, type='retweeted')
        followers = api.followers(user_id)
        for follower in followers:
            follower_id = follower.id_str
            G.add_node(follower_id)
            G.add_edge(follower_id, user_id, type='followed')

    return G






# Preprocess Pheme dataset
pheme_df = pd.read_csv('pheme.csv')
pheme_df['text'] = pheme_df['text'].apply(preprocess_text)
pheme_df.to_csv('preprocessed_pheme.csv', index=False)

# Preprocess Antivax dataset
antivax_df = pd.read_csv('antivax.csv')
antivax_df['text'] = antivax_df['text'].apply(preprocess_text)
antivax_df.to_csv('preprocessed_antivax.csv', index=False)

# Preprocess Constraint dataset
constraint_df = pd.read_csv('constraint.csv')
constraint_df['text'] = constraint_df['text'].apply(preprocess_text)
constraint_df.to_csv('preprocessed_constraint.csv', index=False)

# Preprocess Fakenewsnet dataset
fakenewsnet_df = pd.read_csv('fakenewsnet.csv')
fakenewsnet_df['text'] = fakenewsnet_df['text'].apply(preprocess_text)
fakenewsnet_df.to_csv('preprocessed_fakenewsnet.csv', index=False)
