import pandas as pd 
import numpy as np 
import streamlit as st
from textblob import TextBlob 

st.header("SENTIMENT ANALYSIS MODEL")

def user_input():
    statement = st.text_area("Input statement to be analyzed",placeholder='Start typing')

    dat = {'Statement':statement}
    retdat = pd.DataFrame(dat, index=[0])
    return retdat

stat = user_input()
sub_score = stat['Statement'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
pol_score = stat['Statement'].apply(lambda x: TextBlob(x).sentiment.polarity)

analysis1 = sub_score.apply(lambda x: 'Personal opinion' if x>0.5 else 'Factual statement')

def check_sent(pol_score):
    if pol_score<0:
        return 'Negative statement'
    elif pol_score>0:
        return 'Positive statement'
    else:
        return 'Neutral statement'

analysis = pol_score.apply(check_sent)

st.write('\n')
st.write('\n')
if st.button("Analyze statement"):
    data = {'Subjectivity score':sub_score,
                'Polarity score':pol_score,
                'Subjectivity analysis': analysis1,
                'Sentiment Analysis':analysis
                }
    features = pd.DataFrame(data, index=[0])

    st.write(features)