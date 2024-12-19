import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Function to fetch Reddit posts using libreddit
def fetch_reddit_posts(subreddit, limit=10):
    url = f"https://libreddit.cachyos.org/r/{subreddit}/top"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    posts = []

    for post in soup.find_all('div', class_='post')[:limit]:
        title = post.find('h3').text
        score = post.find('span', class_='score').text
        posts.append({'title': title, 'score': int(score.replace("k", "000").replace(".", ""))})

    return pd.DataFrame(posts)

# Sentiment Analysis Function
def apply_sentiment_analysis(df):
    # Preprocess text (e.g., cleaning if necessary)
    df['clean_title'] = df['title']

    # VADER Sentiment Scores
    df['vader_sentiment'] = df['clean_title'].apply(lambda x: sid.polarity_scores(x)['compound'])

    # TextBlob Sentiment Scores
    df['textblob_sentiment'] = df['clean_title'].apply(lambda x: TextBlob(x).sentiment.polarity)

    return df

# Fetch data and apply sentiment analysis
reddit_df = fetch_reddit_posts("stocks", limit=10)
reddit_df = apply_sentiment_analysis(reddit_df)

print(reddit_df[['title', 'score', 'vader_sentiment', 'textblob_sentiment']])
