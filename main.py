import nltk
import numpy as np
from nltk.corpus import twitter_samples
from process_tweet import process_tweet
from build_freqs import build_freq
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

nltk.download('twitter_samples')
nltk.download('stopwords')

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

# concatenate the lists, 1st part is the positive tweets followed by the negative
tweets = all_positive_tweets + all_negative_tweets

# let's see how many tweets we have
print("Number of tweets: ", len(tweets))

labels = np.append(np.ones((len(all_positive_tweets))), np.zeros((len(all_negative_tweets))))

print(labels.shape)


freqs = build_freq(tweets, labels)

# check data type
print(f'type(freqs) = {type(freqs)}')

# check length of the dictionary
print(f'len(freqs) = {len(freqs)}')






