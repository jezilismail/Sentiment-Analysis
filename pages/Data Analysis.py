import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
from wordcloud import WordCloud


st.title("Data Analysis")
st.markdown("""
            The model was trained on [**Sentiment Analysis for Mental Health**](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data) 
            dataset from Kaggle. 
            This richly curated dataset contains over **52,000** 
            unique textual statements—each annotated with a specific 
            mental health status.

            Classification Labels
            - Normal
            - Anxiety
            - Depression
            - Suicidal
            - Stress
            - Bipolar
            - Personality disorder
            """)

st.markdown('## Data Preview')
df = pd.read_csv('assets/emotion_data.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.dropna(inplace=True)
df.rename(columns={'statement': 'text', 'status': 'label'}, inplace=True)
st.dataframe(df.sample(10, random_state=1).reset_index(drop=True), use_container_width=True)

# Total records
st.markdown(
    f"<h3 style='font-size:28px;'>Total Records: <span style='color:#4CAF50;'>{len(df)}</span></h3>",
    unsafe_allow_html=True
)

# Data distribution plots
st.markdown(
    "<h3 style='font-size:28px;'>Data Distribution</h3>",
    unsafe_allow_html=True
)
sentiment_distribution = df['label'].value_counts()

# Pie chart
fig1, ax1 = plt.subplots()
ax1.pie(sentiment_distribution, labels=sentiment_distribution.index, autopct='%1.1f%%')
ax1.set_title("Distribution of Sentiment Labels in the Dataset", fontsize=14)
st.pyplot(fig1)

# Histogram
fig2, ax2 = plt.subplots()
df.groupby('label').size().plot(kind='barh', color=sns.color_palette('Dark2'), ax=ax2)
ax2.spines[['top', 'right']].set_visible(False)
ax2.set_title("Frequency of Each Sentiment Label", fontsize=14)
st.pyplot(fig2)

# Wordclouds
figures = []
labels = sentiment_distribution.index


for sentiment in labels:
  texts = df[df['label'] == sentiment]
  txt = ' '.join(text.lower() for text in texts['text'])
  wordcloud = WordCloud().generate(txt)
  fig, ax = plt.subplots(figsize=(5, 3))
  ax.imshow(wordcloud, interpolation='bilinear')
  ax.axis('off')
  ax.set_title('{}'.format(sentiment.capitalize()))
  figures.append(fig)

# Display in 2xN grid
st.markdown("## WordClouds by Sentiment Label")
st.markdown("""
Each word cloud visualizes the most frequent words used in texts labeled with a specific sentiment.  
Larger words appear more often in that sentiment category, helping reveal patterns or themes commonly expressed by users in that emotional state.
""")


# Create two columns
cols = st.columns(2)

for i, fig in enumerate(figures):
  with cols[i % 2]:  # alternate between two columns
    st.pyplot(fig)