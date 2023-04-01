import tweepy
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from difflib import SequenceMatcher


consumer_key = 'XXXXXXXXXXXXXXXXXXX'
consumer_secret = 'XXXXXXXXXXXXXXXXXXX'
access_token = 'XXXXXXXXXXXXXXXXXXX'
access_token_secret = 'XXXXXXXXXXXXXXXXXXX'

# create OAuthHandler object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
# set access token and secret
auth.set_access_token(access_token, access_token_secret)
# create tweepy API object to fetch tweets
api = tweepy.API(auth)

def remove_links_and_special_chars(text):
    # Remove links
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # Remove emojis and special characters
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text.lower().strip()

def is_similar(a, b, threshold=0.95):
    return SequenceMatcher(None, a, b).ratio() > threshold


# Defining Search keyword and number of tweets and searching tweets
query = 'biden 2024 -filter:retweets'
max_tweets = 500

seen_tweet_texts = set()
searched_tweets = []
for status in tweepy.Cursor(api.search_tweets, q=query, tweet_mode="extended", lang="en", result_type="mixed").items():
  preprocessed_text = remove_links_and_special_chars(status.full_text)
    # Check if the tweet text is not in the seen_tweet_texts set
  is_duplicate = False
  for seen_text in seen_tweet_texts:
    if is_similar(preprocessed_text, seen_text):
        is_duplicate = True
        break

  if not is_duplicate:
    # Add the preprocessed tweet text to the seen_tweet_texts set and append the tweet to the searched_tweets list
    seen_tweet_texts.add(preprocessed_text)
    searched_tweets.append(status)

    # Break the loop if the desired number of tweets is reached
    if len(searched_tweets) >= max_tweets:
        break

# searched_tweets = [status for status in tweepy.Cursor(api.search_tweets, q=query, tweet_mode="extended", lang="en", result_type="mixed").items(max_tweets)]

#
analyzer = SentimentIntensityAnalyzer()
pos = 0
neg = 0
neu = 0
for tweet in searched_tweets:
  analysis = analyzer.polarity_scores(tweet.full_text)
  if analysis['compound'] > 0:
    pos += 1
  elif analysis['compound'] < 0:
    neg += 1  
  else:
    neu += 1

print("Total Positive = ", pos)
print("Total Negative = ", neg)
print("Total Neutral = ", neu)

#Plotting sentiments
labels = 'Positive', 'Negative', 'Neutral'
sizes = [pos, neg, neu]
colors = ['gold', 'yellowgreen', 'lightcoral']
explode = (0.1, 0, 0)  # explode 1st slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

#Part-3: Creating Dataframe of Tweets
#Cleaning searched tweets and converting into Dataframe
my_list_of_dicts = []
for each_json_tweet in searched_tweets:
    my_list_of_dicts.append(each_json_tweet._json)
    
with open('tweet_json_Data.txt', 'w') as file:
        file.write(json.dumps(my_list_of_dicts, indent=4))
        
my_demo_list = []

#Removing @ handle
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt 

with open('tweet_json_Data.txt', encoding='utf-8') as json_file:  
    all_data = json.load(json_file)
    for each_dictionary in all_data:
        tweet_id = each_dictionary['id']
        text = remove_pattern(each_dictionary['full_text'], "@[\w]*")
        retweet_count = each_dictionary['retweet_count']
        created_at = each_dictionary['created_at']
        my_demo_list.append({'tweet_id': str(tweet_id),
                             'text': str(text),
                             'retweet_count': int(retweet_count),
                             'created_at': created_at,
                            })
        
        tweet_dataset = pd.DataFrame(my_demo_list, columns = 
                                  ['tweet_id', 'text', 
                                   'retweet_count', 
                                   'created_at'])
    
#Writing tweet dataset ti csv file for future reference
tweet_dataset.to_csv('tweet_data.csv')

tweet_dataset.shape
tweet_dataset.head()



tweet_dataset['text'] = np.vectorize(remove_pattern)(tweet_dataset['text'], "@[\w]*")

tweet_dataset.head()

tweet_dataset['text'].head(10)