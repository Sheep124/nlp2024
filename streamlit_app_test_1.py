import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator

data = pd.read_csv('clean_dataset.csv')
prodata = pd.read_csv('processed_data.csv')
prodata.dropna(subset=['lemma_text', 'Sentiment_Rating'], inplace=True)

selected_column = ['lemma_text','Sentiment_Rating']
prodata.columns

st.write('Original Text')
data[['Product Review']]

st.write('Translate Text')
prodata[['translated_text','Sentiment_Rating']]

st.write('Tokenise Text')
prodata[['tokenised_text','Sentiment_Rating']]

st.write('Normalise Text')
prodata[['normalise_text','Sentiment_Rating']]

st.write('Stopword Removal')
prodata[['stopword_text','Sentiment_Rating']]

st.write('Stemming & Lemmatisation')
prodata[['stemming_text','Sentiment_Rating',lemma_text]]

st.write('Cleaned Data')
prodata[selected_column]






finalData = pd.read_csv('Contraction Review & Sentiment Rating.csv')

# Assuming 'x_train' is your training data
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
x = tfidf_vectorizer.fit_transform(prodata['lemma_text'])
y = prodata['Sentiment_Rating']

x_train_tfidf, x_test_tfidf, y_train, y_test = train_test_split(x,y , test_size=0.2, random_state=42)

rf_classifier_tfidf = RandomForestClassifier(**{'max_depth': 80, 'max_features': 'sqrt', 'n_estimators': 100})

# Initialize the translator object
translator = Translator()




# Fit the classifier on the training data
rf_classifier_tfidf.fit(x_train_tfidf, y_train)
# Predict on the testing data
rf_predictions_tfidf = rf_classifier_tfidf.predict(x_test_tfidf)

# Function to analyze sentiment
def analyze_sentiment(review):
    if review:
        # Translate the review to Malay
        translation = translator.translate(review, dest='ms')
        translated_review = translation.text

        # Vectorize the translated review
        review_vectorized = tfidf_vectorizer.transform([translated_review])

        # Predict sentiment and get probability estimates
        sentiment = rf_classifier_tfidf.predict(review_vectorized)[0]
        confidence = rf_classifier_tfidf.predict_proba(review_vectorized)
        confidence_level = max(confidence[0]) * 100

        return sentiment, confidence_level
    else:
        return "Please enter a review.", None

# Streamlit app
def main():
    st.title("Sentiment Analysis")

    # Input text area for entering review
    review = st.text_area("Enter Review:", height=100)

    # Analyze button to perform sentiment analysis
    if st.button("Analyze"):
        sentiment, confidence_level = analyze_sentiment(review)
        
        # Display sentiment result
        if sentiment:
            if sentiment == 'Positive':
                st.success(f"It is a {sentiment} review with {confidence_level:.2f}% confidence.")
            else:
                st.error(f"It is a {sentiment} review with {confidence_level:.2f}% confidence.")
        else:
            st.warning("Please enter a review.")

if __name__ == "__main__":
    main()
