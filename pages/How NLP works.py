import streamlit as st
import pandas as pd
import math
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
nltk.download('punkt_tab')

def clean_text(text):
  text = text.lower()
  text = re.sub(r"[^a-zA-Z]" , " " , text)
  text = re.sub(r'[\?\.\!]+(?=[\?\.\!])', '', text)
  text = re.sub(r'(.)\1+', r'\1\1', text)

  return text

def remove_stopwords(text):
  sw = pickle.load(open('assets/stopwords.sav', 'rb'))
  text = text.split()
  text = [txt for txt in text if txt not in sw]
  text = ' '.join(text)

  return text

def stem(text):
  stemmer = PorterStemmer()
  tokens = word_tokenize(text)
  words = [stemmer.stem(word) for word in tokens]
  statement = " ".join(words)

  return statement

def tokenize(text):
  tokens = word_tokenize(text)

  return tokens

def get_vectorized(corpus):
  df = pd.DataFrame(corpus, columns=["Text", "Sentiment", "Tokens"])
  tokens = [token for text in df['Text'] for token in word_tokenize(text)]
  
  vect_df = []
  for token in tokens:
    tf_s1 = df.iloc[0]["Text"].count(token)
    tf_s2 = df.iloc[1]["Text"].count(token)
    tf_s3 = df.iloc[2]["Text"].count(token)
    tf_s4 = df.iloc[3]["Text"].count(token)
    tf_s1_n = tf_s1 / len(df.iloc[0]["Text"].split()) if tf_s1 else 0
    tf_s2_n = tf_s2 / len(df.iloc[1]["Text"].split()) if tf_s2 else 0
    tf_s3_n = tf_s3 / len(df.iloc[2]["Text"].split()) if tf_s3 else 0
    tf_s4_n = tf_s4 / len(df.iloc[3]["Text"].split()) if tf_s4 else 0
    tf_s1_s = '{}/{}'.format(tf_s1, len(df.iloc[0]["Text"].split())) if tf_s1 else '0'
    tf_s2_s = '{}/{}'.format(tf_s2, len(df.iloc[1]["Text"].split())) if tf_s2 else '0'
    tf_s3_s = '{}/{}'.format(tf_s3, len(df.iloc[2]["Text"].split())) if tf_s3 else '0'
    tf_s4_s = '{}/{}'.format(tf_s4, len(df.iloc[3]["Text"].split())) if tf_s4 else '0'
    idf = math.log(4 / sum([tf_s1, tf_s2, tf_s3, tf_s4]))
    tf_idf_s1 = tf_s1_n * idf
    tf_idf_s2 = tf_s2_n * idf
    tf_idf_s3 = tf_s3_n * idf
    tf_idf_s4 = tf_s4_n * idf

    vect_df.append((
      token,
      tf_s1_s,
      tf_s2_s,
      tf_s3_s,
      tf_s4_s,
      round(idf, 3),
      round(tf_idf_s1, 3),
      round(tf_idf_s2, 3),
      round(tf_idf_s3, 3),
      round(tf_idf_s4, 3)
    ))
  
  return vect_df

st.title("How NLP Works (the gist...)")

st.write("""
This app demonstrates how sentiment analysis works using a machine learning pipeline.

You start by entering labeled text samples, which are added to a dataset (corpus). The text then goes through several preprocessing steps:

1. **Normalization** – Lowercasing and cleaning the text  
2. **Stopword Removal** – Removing common uninformative words  
3. **Stemming** – Reducing words to their root form  
4. **Tokenization** – Splitting text into individual words  
5. **TF-IDF Vectorization** – Converting words into numerical features based on importance

After preprocessing, a **Logistic Regression** model is trained on this vectorized data to classify new inputs as **Positive** or **Negative** based on learned word patterns.
""")

corpus = []
st.markdown("## Enter 4 Text Inputs with Sentiment Labels")
st.markdown("Please provide two inputs each for Positive and Negative to keep the dataset balanced")
st.markdown("These inputs will be added to the dataset for training a sentiment classification model.")
st.markdown("Word limit: 5")

if "corpus" not in st.session_state:
  st.session_state.corpus = []

current_inputs = []

for i in range(4):
  col1, col2 = st.columns([3, 2], gap="medium")
  with col1:
    text = st.text_input(f"Text {i+1}", key=f"text_{i}")
    word_count = len(text.strip().split())
    if word_count > 5:
      st.warning(f"Text {i+1}: Maximum of 5 words allowed.")
      text = ""

  with col2:
    st.markdown("<div style='padding-top: 24px'></div>", unsafe_allow_html=True)
    sentiment = st.radio(
      "Label",
      ["Positive", "Negative"],
      key=f"sentiment_{i}",
      label_visibility="collapsed",
      horizontal=True
    )

  if text:
    current_inputs.append((text, sentiment))

if current_inputs:
    st.session_state.corpus = current_inputs

if st.session_state.corpus:
  df_corpus = pd.DataFrame(st.session_state.corpus, columns=["Text", "Sentiment"])
  st.markdown("### Corpus:")
  st.markdown("A corpus is a collection of text data used for natural language processing tasks.")
  st.dataframe(df_corpus, use_container_width=True)

  st.markdown("### Normalize Text")
  st.markdown("Text normalization prepares raw text by converting it to lowercase and removing unwanted characters, making it cleaner for further processing.")
  if st.button('Start Normalizing Text'):
    st.session_state.normalize = True

  if st.session_state.get("normalize"):
    st.markdown("#### Cleaned Text:")
    df_corpus['Text'] = df_corpus['Text'].apply(clean_text)
    st.session_state.corpus = df_corpus.values.tolist()
    st.dataframe(df_corpus, use_container_width=True)
    words = [word for text in df_corpus['Text'] for word in text.split()]

    st.markdown("#### Remove Stopwords:")
    st.markdown("This step removes common words like 'is', 'the', 'and', which don't carry significant meaning. It helps the model focus on informative words.")
    if st.button('Remove Stopwords'):
      st.session_state.rmv_stopwrd = True
    if st.session_state.get("rmv_stopwrd"):
      df_corpus['Text'] = df_corpus['Text'].apply(remove_stopwords)
      st.session_state.corpus = df_corpus.values.tolist()
      st.dataframe(df_corpus, use_container_width=True)
      
      st.markdown("#### Stem Documents:")
      st.markdown("Stemming reduces words to their base/root form. For example, 'frustrating' becomes 'frustrat'. This helps group similar word forms together.")
      if st.button('Stem (Porter Stemmer)'):
        st.session_state.stem = True
      if st.session_state.get("stem"):
        df_corpus['Text'] = df_corpus['Text'].apply(stem)
        st.session_state.corpus = df_corpus.values.tolist()
        st.dataframe(df_corpus, use_container_width=True)
        
        st.markdown("#### Tokenize Documents:")
        st.markdown("Tokenization splits each sentence into individual words or tokens. This is an essential step for converting text into a format that machine learning models can work with.")
        if st.button('Tokenize'):
          st.session_state.tokenize = True
        if st.session_state.get("tokenize"):
          df_corpus['Tokens'] = df_corpus['Text'].apply(tokenize)
          st.session_state.corpus = df_corpus.values.tolist()
          st.dataframe(df_corpus, use_container_width=True)

          st.markdown("### Vectorize Corpus")
          if st.button('Vectorize (TF-IDF)'):
            st.session_state.vectorize = True

          if st.session_state.get("vectorize"):
            get_vectorized(st.session_state.corpus)
            vect_df = pd.DataFrame(get_vectorized(st.session_state.corpus), columns=['Tokens', 'TF(s1)', 'TF(s2)', 'TF(s3)', 'TF(s4)',
                                            'IDF [log(TD/AD)]', 'TF-IDF(s1)', 'TF-IDF(s2)',
                                            'TF-IDF(s3)', 'TF-IDF(s4)'])
            vect_df.drop_duplicates(inplace=True)
            st.dataframe(vect_df, use_container_width=True, hide_index=True)
            st.markdown("""
              This table shows the Term Frequency-Inverse Document Frequency (TF-IDF) calculation for each token in the corpus.

              - **TF(s1) to TF(s4):** These columns show the term frequency of each token in sentence 1 to 4. For example, a value like `1/2` means the word appears once in a sentence of two words.
                
              - **IDF [log(TD/AD)]:** This is the Inverse Document Frequency. It measures how rare a token is across all documents. Calculated as:
                
                \n&nbsp;&nbsp;&nbsp;&nbsp;**IDF = log(Total Documents / Documents Containing the Token)**

                A higher value means the token is more unique or rare in the dataset.

              - **TF-IDF(s1) to TF-IDF(s4):** These columns contain the final TF-IDF scores for each token in each sentence. TF-IDF is calculated by multiplying TF and IDF:
                
                \n&nbsp;&nbsp;&nbsp;&nbsp;**TF-IDF = TF × IDF**

              These values reflect the importance of a word in a specific sentence, relative to how often it appears across the entire dataset.
              """)
            st.markdown("#### Vectorized Corpus:")
            st.markdown("""
              This is the final vectorized representation of your dataset. Each **row** corresponds to a sentence/document, and each **column** represents a token from the corpus.

              - The numbers in the cells are the **TF-IDF scores** of each token in that sentence.
              - A score of `0` means the token doesn't appear in that sentence.
              - The **Label** column is the target sentiment assigned to each row (Positive or Negative).

              This table is the actual input used by machine learning models like Logistic Regression for training or prediction. Each sentence is now a numeric feature vector.
              """)
            tfidf_matrix = vect_df[['TF-IDF(s1)', 'TF-IDF(s2)', 'TF-IDF(s3)', 'TF-IDF(s4)']].transpose()
            tfidf_matrix.columns = vect_df['Tokens'].values
            tfidf_matrix['Label'] = list(df_corpus['Sentiment'])
            st.dataframe(tfidf_matrix, use_container_width=True, hide_index=True)

            st.markdown("#### Train Model:")
            st.markdown("""
              **Logistic Regression** is a simple and effective machine learning model used for classification tasks like sentiment analysis.  
              It learns patterns in your training data and uses those patterns to predict whether new input is Positive or Negative based on the words it contains.
              """)
            if st.button('Use LogisticRegression'):
              st.session_state.LogReg = True
            if st.session_state.get("LogReg"):
              encoder = LabelEncoder()
              vectorizer = TfidfVectorizer()
              model = LogisticRegression(max_iter=1000)
              X = vectorizer.fit_transform(df_corpus['Text'])            
              y = df_corpus['Sentiment']
              y = encoder.fit_transform(y)
              model.fit(X, y)
              st.markdown("Training completed")

              st.markdown("#### Test Your Model")

              st.markdown("""
              <style>
              .badge-box {
                  display: flex;
                  flex-wrap: wrap;
                  gap: 10px;
                  padding: 15px;
                  border: 1px solid #555;
                  border-radius: 10px;
                  background-color: #111;
                  margin: 0 0 24px 0;
              }
              .badge {
                  background-color: #03fcf0;
                  color: black;
                  padding: 6px 12px;
                  border-radius: 25px;
                  font-size: 14px;
                  font-family: sans-serif;
              }
              </style>
              """, unsafe_allow_html=True)

              st.markdown('Try to mostly use words from the documents you gave inorder to get good prediction, as our traing data is very small')
              st.markdown('Use these words:')
              st.markdown('<div class="badge-box">' + ''.join([f'<div class="badge">{word}</div>' for word in words]) + '</div>', unsafe_allow_html=True)
              col1, col2 = st.columns([4, 1], gap='medium')
              with col1:
                text = st.text_input('Enter text to analyse sentiment:')
              with col2:
                st.markdown("<div style='padding-top: 28px'></div>", unsafe_allow_html=True)
                if st.button('Analyse'):
                  st.session_state.analyse = True
              if st.session_state.get("analyse"):
                text = clean_text(text)
                text = remove_stopwords(text)
                text = stem(text)
                sentiment = encoder.inverse_transform(model.predict(vectorizer.transform([text])))
                st.success('{}'.format(sentiment[0]))
      
              