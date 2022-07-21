
import streamlit as st
import pickle
import tweepy
import pandas as pd
import string
import re
import nltk
def create_features_from_df(df):
    stopword = nltk.corpus.stopwords.words('english')

    def remove_punct(text):
      text  = "".join([char for char in text if char not in string.punctuation])
      return text

    def clean_text(text):
      txt = re.sub("[( ' )( ')(' )]", ' ', text)
      txt=re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", txt)
      return txt.lower()

    def remove_stopwords(text):
      text  = " ".join([word for word in text.split(" ") if word not in stopword])
      return text

    df['new_tweets'] = df['tweets'].apply(lambda x: remove_punct(str(x)))
    df['new_tweets'] = df['new_tweets'].apply(lambda x: clean_text(str(x)))
    df['new_tweets'] = df['new_tweets'].apply(lambda x: remove_stopwords(str(x)))
    df.dropna()
    return df
def get_tweets(item):
    #Twitter API credentials
    
    consumer_key =  "2y5779N6k5EZpOz3VmOyabJHc"
    consumer_secret = "q4DR8al72steNMS8Uf4PMmAU9sR0OTWMEwbu2DGZU35S7jN1ff"
    access_token = "1534057526483832832-cHWlsx6qpL9XlXt5zCSbG1pIom6lom"
    access_token_secret = "Kj7bDn8dlvQEloRflgpAtLC1IgOoF9dkWFluRfux5MlzD"
    
    OAUTH_KEYS = {'consumer_key':consumer_key, 'consumer_secret':consumer_secret,
    'access_token_key':access_token, 'access_token_secret':access_token_secret}
    auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # Request
    search = tweepy.Cursor(api.search_tweets, q=item).items(60)

    # Creation des listes pour chaque tweet
    sn = []
    text = []
    timestamp =[]
    for tweet in search:
        
        sn.append(tweet.user.screen_name)
        text.append(tweet.text)
        
    # df_features
    df_features = pd.DataFrame(
         {'tweets': text 
        })
    # df_show_info
    
    
    df_show_info = pd.DataFrame(

        {'User Screen Name': sn
        })
    
    return (df_features,df_show_info)
def get_category_name(category_id):
    category_codes = {
        'Real': 0,
        'Fake': 1
    }
    for category, id_ in category_codes.items():    
        if id_ == category_id:
            return category
def predict_from_features(features):
    svc_model = pickle.load(open("svc_model.sav", 'rb'))
    predictions_pre = svc_model.predict(features)

    predictions = []

    for cat in predictions_pre:
           predictions.append(cat)

    categories = [get_category_name(x) for x in predictions]
    return categories
def complete_df(df, categories):
    df['Prediction'] = categories
    return df

def main():
    st.title('Tweeter Fake and Real Prediction')
    search=st.text_input('Enter the Key')


   
    
    
    if st.button("SUBMIT"):
        try:
            df_features,df_show_info = get_tweets(search)
           
            #cleaning_data = pickle.load(open("cleaning_data.sav", 'rb'),df_features)
            df_features = create_features_from_df(df_features)
            tfidf_vectorizer= pickle.load(open("tfidf_vectorizer.sav", 'rb'))
            features = tfidf_vectorizer.transform(df_features['new_tweets']).toarray()
            predictions = predict_from_features(features)
            df = complete_df(df_features, predictions)
            
            
            st.table(df.head(10))
        except:
            st.warning('Input valid  Keyname')
if __name__=='__main__':
    main()