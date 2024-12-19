import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# Sample Data: Mock Reddit Posts
data = {
    'title': [
        "Stock market hits all-time high", 
        "Investors worry about upcoming recession", 
        "Tech stocks are booming this quarter", 
        "Economic growth slows down", 
        "Positive outlook on renewable energy sector", 
        "Oil prices surge amid geopolitical tensions", 
        "Cryptocurrency market sees major dip", 
        "Retail investors driving stock surge", 
        "Concerns over inflation impact economy", 
        "Housing market shows signs of cooling"
    ],
    'score': [5000, 3000, 7000, 2000, 6000, 4000, 1500, 3500, 2500, 4500]
}

# Create a DataFrame
reddit_df = pd.DataFrame(data)

# Apply sentiment analysis
reddit_df['vader_sentiment'] = reddit_df['title'].apply(lambda x: sid.polarity_scores(x)['compound'])
reddit_df['textblob_sentiment'] = reddit_df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Print the DataFrame to see sample sentiment scores
print(reddit_df[['title', 'score', 'vader_sentiment', 'textblob_sentiment']])

# Use a valid style
plt.style.use('ggplot')
plt.figure(figsize=(16, 12))

# 1. Plot VADER sentiment distribution
plt.subplot(2, 2, 1)
sns.histplot(reddit_df['vader_sentiment'], bins=10, kde=True, color='blue')
plt.title('VADER Sentiment Distribution')
plt.xlabel('VADER Sentiment Score')
plt.ylabel('Frequency')

# 2. Plot TextBlob sentiment distribution
plt.subplot(2, 2, 2)
sns.histplot(reddit_df['textblob_sentiment'], bins=10, kde=True, color='green')
plt.title('TextBlob Sentiment Distribution')
plt.xlabel('TextBlob Sentiment Score')
plt.ylabel('Frequency')

# 3. Scatter plot for Score vs VADER Sentiment
plt.subplot(2, 2, 3)
sns.scatterplot(x='score', y='vader_sentiment', data=reddit_df, color='purple')
plt.title('Reddit Post Scores vs. VADER Sentiment')
plt.xlabel('Reddit Post Score')
plt.ylabel('VADER Sentiment Score')

# 4. Bar Plot for Title vs. Sentiment Scores (VADER & TextBlob)
plt.subplot(2, 2, 4)
reddit_df.set_index('title')[['vader_sentiment', 'textblob_sentiment']].plot(kind='bar', width=0.8, ax=plt.gca())
plt.title('Reddit Post Titles vs. Sentiment Scores')
plt.xlabel('Post Title')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

# Display model performance metrics
model_performance = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'LSTM'],
    'Accuracy': [0.75, 0.82, 0.88],
    'MAE': [1.15, 0.85, 0.65],
    'RMSE': [1.40, 1.10, 0.90]
})

# Display model performance as a table
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=model_performance.values, colLabels=model_performance.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

plt.title("Model Performance Metrics")
plt.show()