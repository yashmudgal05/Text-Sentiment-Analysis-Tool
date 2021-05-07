import streamlit as st
import json
import pickle
import re
import tweepy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

import tkinter as tk
from tkinter import filedialog

from google_trans_new import google_translator 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
from tensorflow.keras.models import load_model

import csv
import io
from selenium import webdriver
from selenium.common import exceptions
import time

import preprocess_yash as kgp


consumerKey = 'ovfz836QOdYr2qzyCniD1gw3M'
consumerSecret = 'cxisPr0a0TX4pLMvlx9DPRPtQ1vyEALpeyPGy5004AtGurYBS1'
accessToken = '1165486442019078144-TO3j66TDxjWUzddrPqYF4BelVTTrUE'
accessTokenSecret = 'lTDx5j1v4y11gJZ2Hw5dc4bdv0u1LY1NOO1wO6BS8QO4e'

authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret) 
authenticate.set_access_token(accessToken, accessTokenSecret) 
api = tweepy.API(authenticate, wait_on_rate_limit = True)


def main():
    
    activities = ["Home", "Twitter sentiment analysis tool", "Hate speech detection", "Emotion recogination", "Youtube scraping", "Whatsapp"]
    
    choice = st.sidebar.selectbox("Select Your Activity",activities)
    
    if choice == "Home":
        st.title("🔥 TEXT Sentiment analysis tool 🔥")
        
    elif choice == "Twitter sentiment analysis tool":
    
        st.title("🔥 Tweets sentiment analysis tool 🔥")
        
        st.subheader("1. Do you have a tweet & you you want to analyze it?")
        
        raw_text = st.text_area("Enter the exact tweet, we will handle the text prepocessing")
        
        Analyzer_choice = st.selectbox("Select the language of the tweet",  ["English","Hindi", "Hinglish"])
        
        analyze_1 = st.button("Analyze", key='analyze_1')
        
        if analyze_1:
            
            if Analyzer_choice == "English":
            
                processed_text = get_clean(raw_text)
                emotion = twitter_clf.predict(twitter_tfidf.transform([processed_text]))
                
                st.success("Analyzing the tweet")
                
                if emotion[0] == 0:
                    st.write("The tweet contains **NEGATIVE** emotion")
                elif emotion[0] == 4:
                    st.write("The tweet contains **POSITIVE** emotion")
                    
            if Analyzer_choice == "Hindi":
                
                processed_text = get_clean(raw_text)
                
                translator = google_translator()  
                processed_text = translator.translate(processed_text, lang_src='hi', lang_tgt='en')
                
                emotion = twitter_clf.predict(twitter_tfidf.transform([processed_text]))
                
                st.success("Analyzing the tweet")
                
                if emotion[0] == 0:
                    st.write("The tweet contains **NEGATIVE** emotion")
                elif emotion[0] == 4:
                    st.write("The tweet contains **POSITIVE** emotion")
                    
            if Analyzer_choice == "Hinglish":
                
                processed_text = get_clean(raw_text)
                
                translator = google_translator()  
                processed_text = translator.translate(processed_text, lang_src='hi', lang_tgt='en')
                
                emotion = twitter_clf.predict(twitter_tfidf.transform([processed_text]))
                
                st.success("Analyzing the tweet")
                
                if emotion[0] == 0:
                    st.write("The tweet contains **NEGATIVE** emotion")
                elif emotion[0] == 4:
                    st.write("The tweet contains **POSITIVE** emotion")
                
                
                
        st.subheader("2. Analyzing a twitter handle")
        
        raw_text = st.text_input("Enter the exact twitter handle name (e.g. @user_name)")
        raw_text = raw_text[1:]
        numTweet = st.number_input("Enter number of tweets you want to analyze", key='numTweet2')
        
        analyze_2 = st.button("Analyze", key='analyze_2')
        
        if analyze_2:
            
            df = Plot_Analysis(raw_text, numTweet)
            df.to_csv("handle_analysis.csv")
            
            st.write(sns.countplot(x=df["Analysis"], data=df))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(use_container_width=False)
 
            plt.figure(figsize =(4, 3))
            st.write(df["Analysis"].value_counts().plot.pie(autopct='%1.2f%%'))
            st.pyplot(plt.show(), use_container_width=False)
            
        
        st.subheader("3. Want to know public sentiments about something ?")
        
        raw_text = st.text_input("Enter the keyword (e.g. vaccine)")
        numTweet = st.number_input("Enter number of tweets you want to analyze", key='numTweet3')
        
        analyze_3 = st.button("Analyze", key='analyze_3')
        
        if analyze_3:
            
            df = keywordAnalysis(get_clean(raw_text), numTweet)
            df.to_csv("keyword_analysis.csv")
            
            st.write(sns.countplot(x=df["Analysis"], data=df))
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(use_container_width=False)
 
            plt.figure(figsize =(4, 3))
            st.write(df["Analysis"].value_counts().plot.pie(autopct='%1.2f%%'))
            st.pyplot(plt.show(), use_container_width=False)
            
            
        st.subheader("4. Want to compare public sentiments about two things or items ?")
        track_keyword_1 = st.text_input("Enter the name of 1st item")
        track_keyword_2 = st.text_input("Enter the name of 2nd item")
        numTweet = st.number_input("Enter number of tweets you want to analyze", key='numTweet4')
        
        analyze_4 = st.button("Analyze", key='analyze_4')
        
        if analyze_4:
            
            df1 = comparisonAnalysis(track_keyword_1, numTweet)
            df2 = comparisonAnalysis(track_keyword_2, numTweet)
            
            global item1
            global item2
            item1 = 0
            item2 = 0
            l1 = []
            l2 = []
            
            data = pd.DataFrame(columns=[track_keyword_1, track_keyword_2])
            
            for item in df1["Sentiment"]:
                if item == 4:
                    item1 += 1
                else:
                    item1 += 0
                l1.append(item1)
            
            for item in df2["Sentiment"]:
                if item == 4:
                    item2 += 1
                else:
                    item2 += 0
                l2.append(item2)
                
            data[track_keyword_1] = l1
            data[track_keyword_2] = l2
            
            data.to_csv('data.csv', index=False)
            
            chart_data = pd.read_csv("data.csv", usecols=[track_keyword_1, track_keyword_2])
            st.line_chart(chart_data)
            
            
        st.subheader("5. Want to compare two twitter handles")
        
        handle_1 = st.text_input("Enter the first twitter handle name (e.g. @user_name)")
        handle_1 = handle_1[1:]
        handle_2 = st.text_input("Enter the second twitter handle name (e.g. @user_name)")
        handle_2 = handle_2[1:]
        numTweet = st.number_input("Enter number of tweets you want to analyze", key='numTweet5')
        
        analyze_5 = st.button("Analyze", key='analyze_5')
        
        if analyze_5:
            
            df1 = handle_analysis(handle_1, numTweet)
            df2 = handle_analysis(handle_2, numTweet)
            
            global item3
            global item4
            item3 = 0
            item4 = 0
            l1 = []
            l2 = []
            
            data = pd.DataFrame(columns=[handle_1, handle_2])
            
            for item in df1["Sentiment"]:
                if item == 4:
                    item3 += 1
                else:
                    item3 += 0
                l1.append(item3)
            
            for item in df2["Sentiment"]:
                if item == 4:
                    item4 += 1
                else:
                    item4 += 0
                l2.append(item4)
                
            data[handle_1] = l1
            data[handle_2] = l2
            
            data.to_csv('handle_data.csv', index=False)
            
            chart_data = pd.read_csv("handle_data.csv", usecols=[handle_1, handle_2])
            st.line_chart(chart_data)
            
    elif choice == "Hate speech detection":
        
        st.title("🔥 Hate speech detection 🔥")
        st.subheader("Classify textual data as:\n1. hate speech\n2. offensive language\n3. neither of the two")
        
        st.subheader("1. For **ENGLISH** language")
        
        raw_text = st.text_area("Enter the speech you want to classify (we will take care of pre-processing the text)", key="hateSpeech1")
        
        analyze_4 = st.button("Analyze", key='analyze_4')
        
        if analyze_4:
        
            speech_emotion = np.argmax(hate_Speech_model.predict(get_encoded(raw_text)), axis=-1)
            
            if speech_emotion == 0:
                st.write("**HATE SPEECH**")
            elif speech_emotion == 1:
                st.write("**OFFENSIVE LANGUAGE**")
            elif speech_emotion == 2:
                st.write("**NEITHER**")
                
        st.subheader("2. For हिंदी  and **HINGLISH** language")
        
        raw_text = st.text_area("Enter the speech you want to classify (we will take care of pre-processing the text)", key="hateSpeech2")
        
        analyze_8 = st.button("Analyze", key='analyze_8')
        
        if analyze_8:
        
            translator = google_translator()  
            raw_text = translator.translate(raw_text, lang_src='hi', lang_tgt='en')  
            
            speech_emotion = np.argmax(hate_Speech_model.predict(get_encoded(raw_text)), axis=-1)
            
            if speech_emotion == 0:
                st.write("**HATE SPEECH**")
            elif speech_emotion == 1:
                st.write("**OFFENSIVE LANGUAGE**")
            elif speech_emotion == 2:
                st.write("**NEITHER**")   
                
    elif choice == "Emotion recogination":
        
        st.title("🔥 Emotion recogination 🔥")
        st.subheader("Classify the emotion of a textual data as:\n1. surprise\n2. sadness\n3. love\n4. anger\n5. joy\n6. fear")
        
        st.subheader("1. For **ENGLISH** language")
        
        raw_text = st.text_area("Enter the text you want to analyze", key="emotion1")
        
        analyze_6 = st.button("Analyze", key='analyze_6')
        
        if analyze_6:
            x = get_clean(raw_text)
            emotion = model_logisticRegression.predict(tfidf_logisticRegression.transform([x]))
            
            if emotion == "SURPRISE":
                st.write("The emotion of text is **SURPRISE**")
            elif emotion == "SADNESS":
                st.write("The emotion of text is **SADNESS**")
            elif emotion == "LOVE":
                st.write("The emotion of text is **LOVE**")
            elif emotion == "ANGER":
                st.write("The emotion of text is **ANGER**")
            elif emotion == "JOY":
                st.write("The emotion of text is **JOY**")
            elif emotion == "FEAR":
                st.write("The emotion of text is **FEAR**")
                
                
        st.subheader("2. For हिंदी  and **HINGLISH** language")
        
        raw_text = st.text_area("Enter the text you want to analyze", key="emotion2")
        
        analyze_7 = st.button("Analyze", key='analyze_7')
        
        if analyze_7:
            x = get_clean(raw_text)
             
            translator = google_translator()  
            x = translator.translate(x, lang_src='hi', lang_tgt='en')

            emotion = model_logisticRegression.predict(tfidf_logisticRegression.transform([x]))
            
            if emotion == "SURPRISE":
                st.write("The emotion of text is **SURPRISE**")
            elif emotion == "SADNESS":
                st.write("The emotion of text is **SADNESS**")
            elif emotion == "LOVE":
                st.write("The emotion of text is **LOVE**")
            elif emotion == "ANGER":
                st.write("The emotion of text is **ANGER**")
            elif emotion == "JOY":
                st.write("The emotion of text is **JOY**")
            elif emotion == "FEAR":
                st.write("The emotion of text is **FEAR**")

    elif choice == "Youtube scraping":
        
        st.title("🔥 Youtube comments scraping 🔥")
        st.subheader("Model will scrap comments from a youtube video & gives overall polarity of the comments")
        st.subheader(' ')
        raw_text = st.text_area("Please paste the link of YOUTUBE video")
        
        analyze_9 = st.button("Scrap", key='analyze_9')
        
        if analyze_9:
            
            title = scrape(raw_text)
            
            st.write("Video's title :point_right: " + ' ' + '**' + title + '** :point_left:')
            
            df = pd.read_csv('youtube_results.csv', encoding="utf-16", sep=',')
            df['Comment'] = df['Comment'].apply(lambda x : get_clean(x))
            
            pos,neg=0,0

            for comment in df['Comment']:
                try:    
                    emotion = clf.predict(tfidf.transform([comment]))
                    
                    if emotion[0] == 0:
                        neg+=1
                    elif emotion[0] == 4:
                        pos+=1
                except:
                    pass
            
            plt.figure(figsize =(2, 2))
            
            neg=abs(neg)
            labels = ['positive','negative']
            sizes = [pos,neg]
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes ,labels=labels, autopct='%1.1f%%')
            plt.title('Polarity Analysis')
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(plt.show(), use_container_width=False)
            
    elif choice == "Whatsapp":
        
        st.title("🔥 Whatsapp chat analysis 🔥")
        st.subheader("Model will gives the polarity of a overall whatsapp chat along with individual participant's chat polarity")
        st.subheader(' ')
        st.write('Please upload the chat file **_.txt file_** ')
        
        analyze_10 = st.button("Upload", key='analyze_10')
        
        if analyze_10:
            root = tk.Tk()
            root.withdraw()
            root.focus_force()
            
            filepath = tk.filedialog.askopenfilename(parent=root)
            st.write('Working on ' + ':point_right: **' + filepath + '** :point_left:')
            
            opinion={}

            df = pd.read_csv(filepath, sep="\t", names = ['text'])
            
            pos,neg=0,0
            
            for line in df['text']:
                try:
                    chat=line.split('-')[1].split(':')[1]
                    name=line.split('-')[1].split(':')[0]
                    
                    translator = google_translator()  
                    chat = translator.translate(chat, lang_src='hi', lang_tgt='en')
                    
                    chat = get_clean(chat)
                    
                    if opinion.get(name,None) is None:
                        opinion[name]=[0,0]
                        
                    emotion = clf.predict(tfidf.transform([chat]))
                    
                    if emotion[0] == 0:
                        res = 'negative'
                    elif emotion[0] == 4:
                        res = 'positive'
                    
                    if res=='positive':
                        pos+=1
                        opinion[name][0]+=1
                    else:
                        neg+=1
                        opinion[name][1]+=1
                except:
                    pass
                
            neg=abs(neg)
            labels = ['positive','negative']
            sizes = [pos,neg]
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes ,labels=labels, autopct='%1.1f%%')
            plt.title('Whatsapp Sentiment Analysis')
            st.pyplot(plt.show(), use_container_width=False)
            
            names,positive,negative=[],[],[]
            for name in opinion:
                names.append(name)
                positive.append(opinion[name][0])
                negative.append(opinion[name][1])
                
            def autolabel(rects):
                for rect in rects:
                    h = rect.get_height()
                    ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                            ha='center', va='bottom')
                    
            ind = np.arange(len(names))
            width=0.3
            max_x=max(max(positive),max(negative))+2
            
            fig = plt.figure()
            ax = fig.add_subplot()
            
            yvals = positive
            rects1 = ax.bar(ind, yvals, width, color='g')
            zvals = negative
            rects2 = ax.bar(ind+width, zvals, width, color='r')
            
            ax.set_xlabel('Names')
            ax.set_ylabel('Sentiment')
            
            ax.set_xticks(ind+width)
            ax.set_yticks(np.arange(0,max_x+20,10))
            ax.set_xticklabels( names )
            ax.legend( (rects1[0], rects2[0]), ('positive', 'negative') )
            ax.set_title('Whatsapp Chat Sentiment Analysis')
            
            autolabel(rects1)
            autolabel(rects2)
            
            st.pyplot(plt.show(), use_container_width=False)


def Plot_Analysis(raw_text, numTweet):
    st.success("Analyzing the twitter handle")
    
    posts = api.user_timeline(screen_name=raw_text, count = numTweet, lang ="en", tweet_mode="extended")
    
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
    
    # Clean the tweets
    df['Tweets'].apply(get_clean)  
    					   		
    # Create two new columns 'Subjectivity' & 'Polarity'
    df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
    df['Polarity'] = df['Tweets'].apply(getPolarity)
    					    
    df['Analysis'] = df['Polarity'].apply(getAnalysis)
    
    return df

def handle_analysis(raw_text, numTweet):
    
    posts = api.user_timeline(screen_name=raw_text, count = numTweet, lang ="en", tweet_mode="extended")
    
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
    
    # Clean the tweets
    df['Tweets'].apply(get_clean)
    
    df['Sentiment'] = df['Tweets'].apply(lambda x: predict_sentiment(x)[0])
    
    return df
         
def get_clean(x):
    x = str(x).lower().replace('\\', ' ').replace('_', ' ').replace('.', ' ')
    x = kgp.remove_rt(x)
    x = kgp.cont_exp(x)
    x = kgp.remove_emails(x)
    x = kgp.remove_urls(x)
    x = kgp.remove_html_tags(x)
    x = kgp.remove_accented_chars(x)
    x = kgp.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    x = kgp.make_base(x)
    return x    

def get_encoded(x):
    max_length = 120
    
    x = get_clean(x)
    
    # convering text data to numerical sequence
    x = hate_speech_token.texts_to_sequences([x])
    
    # fixing input data length
    x = pad_sequences(x, maxlen=max_length, padding='post')
    
    return x

class MyStreamListener(tweepy.StreamListener):
    
    def __init__(self, maxCount, df):
        super().__init__()
        self.max_tweets = maxCount + 2
        self.tweet_count = 0
    
    def on_status(self, status):
        print(status.text)
        
    def on_data(self, data):
        
        raw_twitts = json.loads(data)
        
        try:
            self.tweet_count += 1
            if(self.tweet_count == self.max_tweets):
                return(False)
            else:
                df.loc[len(df.index)] = [str(raw_twitts['text'])]
                return
        except:
            pass
        
    def on_error(self, status_code):
        if status_code == 420:
            print('Error 420')
            #returning False in on_error disconnects the stream
            return False  

def keywordAnalysis(track_keyword, numTweet):
    global df 
    df = pd.DataFrame(columns=['Tweets'])

    myStreamListener = MyStreamListener(numTweet, df)
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)

    myStream.filter(track = [track_keyword])
    
    # Clean the tweets
    df['Tweets'].apply(get_clean)
    
    # Create two new columns 'Subjectivity' & 'Polarity'
    df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
    df['Polarity'] = df['Tweets'].apply(getPolarity)
    
    df['Analysis'] = df['Polarity'].apply(getAnalysis)
    
    return df

def predict_sentiment(x):
        x = [x]
        sent = twitter_clf.predict(twitter_tfidf.transform(x))
        return sent

def comparisonAnalysis(track_keyword, numTweet):
    global df 
    df = pd.DataFrame(columns=['Tweets'])

    myStreamListener = MyStreamListener(numTweet, df)
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)

    myStream.filter(track = [track_keyword])
    
    # Clean the tweets
    df['Tweets'] = df['Tweets'].apply(lambda x: get_clean(x))
    
    df['Sentiment'] = df['Tweets'].apply(lambda x: predict_sentiment(x)[0])
    
    return df

def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity
    					   
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'
        
        
def scrape(url):
    
    option = webdriver.ChromeOptions()
    option.add_argument("--headless")

    driver = webdriver.Chrome('./webdrivers/chromedriver', options=option)

    driver.get(url)
    driver.maximize_window()
    time.sleep(5)

    try:
        title = driver.find_element_by_xpath('//*[@id="container"]/h1/yt-formatted-string').text
        comment_section = driver.find_element_by_xpath('//*[@id="comments"]')
    except exceptions.NoSuchElementException:
        # Note: Youtube may have changed their HTML layouts for
        # videos, so raise an error for sanity sake in case the
        # elements provided cannot be found anymore.
        error = "Error: Double check selector OR "
        error += "element may not yet be on the screen at the time of the find operation"
        print(error)

    # Scroll into view the comment section, then allow some time
    # for everything to be loaded as necessary.
    driver.execute_script("arguments[0].scrollIntoView();", comment_section)
    time.sleep(7)

    # Scroll all the way down to the bottom in order to get all the
    # elements loaded (since Youtube dynamically loads them).
    last_height = driver.execute_script("return document.documentElement.scrollHeight")

    while True:
        # Scroll down 'til "next load".
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")

        # Wait to load everything thus far.
        time.sleep(2)

        # Calculate new scroll height and compare with last scroll height.
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    # One last scroll just in case.
    driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")

    username_elems = []
    comment_elems = []

    try:
        # Extract the elements storing the usernames and comments.
        username_elems = driver.find_elements_by_xpath('//*[@id="author-text"]')
        comment_elems = driver.find_elements_by_xpath('//*[@id="content-text"]')
    except exceptions.NoSuchElementException:
        error = "Error: Double check selector OR "
        error += "element may not yet be on the screen at the time of the find operation"
        print(error)

    with io.open('youtube_results.csv', 'w', newline='', encoding="utf-16") as file:
        writer = csv.writer(file, delimiter =",", quoting=csv.QUOTE_ALL)
        writer.writerow(["Username", "Comment"])
        for username, comment in zip(username_elems, comment_elems):
            writer.writerow([username.text, comment.text])

    driver.close()
    
    return title


if __name__ == "__main__":
    
    twitter_clf = pickle.load(open('twitter_clf.pkl', 'rb'))
    twitter_tfidf = pickle.load(open('twitter_tfidf.pkl', 'rb'))
    
    hate_Speech_model = load_model('hate_Speech_model')
    hate_speech_token = pickle.load(open('hate_Speech_token.pkl', 'rb'))
    
    model_logisticRegression = pickle.load(open('model_emotionRecogination.pkl', 'rb'))
    tfidf_logisticRegression = pickle.load(open('tfidf_emotionRecogination.pkl', 'rb'))
    
    clf = pickle.load(open('sentiment_clf.pkl', 'rb'))
    tfidf = pickle.load(open('sentiment_tfidf.pkl', 'rb'))
    
    main()