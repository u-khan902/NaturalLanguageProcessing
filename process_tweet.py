import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import numpy as np
import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
def process_tweet(tweet):
  stemmer = PorterStemmer()

  stopwords_english = stopwords.words('english')
  tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

  tweet = re.sub(r'^RT[\s]+', '', tweet)
  tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
  tweet = re.sub(r'#', '', tweet)
  tweet = re.sub(r'\$\w*','',tweet)
  tweet_token  = tokenizer.tokenize(tweet)
  for word in tweet_token:
    if word not in stopwords_english and word not in string.punctuation:
      tweet_token.append(stemmer.stem(word))
  return tweet_token
