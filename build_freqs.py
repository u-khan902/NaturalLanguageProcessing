from process_tweet import process_tweet
import numpy as np
def build_freq(tweets, ys):
  ylist = np.squeeze(ys).tolist()
  freq = {}
  for y, tweet in zip(ylist, tweets):
    for word in process_tweet(tweet):
      pair = (word, y)
      if pair in freq:
        freq[pair] += 1
      else:
        freq[pair] = 1
  return freq