import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load Dataset
df = pd.read_csv("twitter_training.csv", header=None)

# Rename Columns
df.columns = ["Tweet_ID", "Topic", "Sentiment", "Text"]

# Display Basic Information
print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nSentiment Counts:")
print(df["Sentiment"].value_counts())

# Remove Missing Values
df.dropna(inplace=True)

# -----------------------------
# Sentiment Distribution
# -----------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Sentiment")
plt.title("Sentiment Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("sentiment_distribution.png")
plt.show()

# -----------------------------
# Positive Word Cloud
# -----------------------------
positive_text = " ".join(
    df[df["Sentiment"] == "Positive"]["Text"].astype(str)
)

wordcloud_positive = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(positive_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Sentiment Word Cloud")
plt.tight_layout()
plt.savefig("wordcloud_positive.png")
plt.show()

# -----------------------------
# Negative Word Cloud
# -----------------------------
negative_text = " ".join(
    df[df["Sentiment"] == "Negative"]["Text"].astype(str)
)

wordcloud_negative = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(negative_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Sentiment Word Cloud")
plt.tight_layout()
plt.savefig("wordcloud_negative.png")
plt.show()

# -----------------------------
# Topic-wise Sentiment Analysis
# -----------------------------
top_topics = df["Topic"].value_counts().head(10).index

topic_df = df[df["Topic"].isin(top_topics)]

plt.figure(figsize=(12, 6))
sns.countplot(
    data=topic_df,
    x="Topic",
    hue="Sentiment"
)
plt.xticks(rotation=45)
plt.title("Top 10 Topics by Sentiment")
plt.tight_layout()
plt.savefig("topic_sentiment_analysis.png")
plt.show()

print("\nFiles Saved Successfully:")
print("1. sentiment_distribution.png")
print("2. wordcloud_positive.png")
print("3. wordcloud_negative.png")
print("4. topic_sentiment_analysis.png")