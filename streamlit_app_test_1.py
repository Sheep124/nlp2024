import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from googletrans import Translator
import emoji
import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from googletrans import Translator
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

normalized_word = pd.read_csv("normalisasi.csv")
normalized_word_dict = {}


lemmatizer = WordNetLemmatizer()

data = pd.read_csv('clean_dataset.csv')
prodata = pd.read_csv('processed_data.csv')
prodata.dropna(subset=['lemma_text', 'Sentiment_Rating'], inplace=True)

def convert_demojis(text):
    text_temp = emoji.demojize(text)
    return text_temp.replace(":", " ")

def remove_punctuation(text):
    punctuation_pattern = r'[^\w\s]'
    text_without_punctuation = (re.sub(punctuation_pattern, ' ', text)).lower()
    
    cleaned_text_temp = text_without_punctuation.replace('\n', ' ')
    
    punctuation_chars = '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
    cleaned_text = ''.join(char for char in cleaned_text_temp if char not in punctuation_chars)
    return cleaned_text

def expand_contractions(text):
    return contractions.fix(text)

def translate_text_deploy(text, source_lang, target_lang):
    try:
        # Initialize the translator
        translator = Translator()

        # Translate text
        translation = translator.translate(text, src=source_lang, dest=target_lang)

        return translation.text
    except Exception as e:
        return str(e)

def remove_stopwords(text):
    # Tokenize the text
    text = str(text)
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    stopwords_removed = [word for word in tokens if word.lower() not in stopwords.words('english')]
    # Join the words back into a single string
    return ' '.join(stopwords_removed)

def word_tokenize_wrapper(text):
    text = str(text)
    return word_tokenize(text)

def normalized_term(document):
    return [normalized_word_dict[term] if term in normalized_word_dict else term for term in document]

def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    do = []
    for w in text:
        dt = stemmer.stem(w)
        do.append(dt)
    d_clean=[]
    d_clean=" ".join(do)
    print(d_clean)
    return d_clean



def lemmatize_text(words):
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

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
prodata[['stemming_text','Sentiment_Rating','lemma_text']]

finalData = pd.read_csv('Contraction Review & Sentiment Rating.csv')

st.write('Cleaned Data')
finalData[selected_column]


# Assuming 'x_train' is your training data
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
x = tfidf_vectorizer.fit_transform(prodata['lemma_text'])
y = prodata['Sentiment_Rating']

x_train_tfidf, x_test_tfidf, y_train, y_test = train_test_split(x,y , test_size=0.2, random_state=42)

#rf_classifier_tfidf = LogisticRegression(**{'C': 1, 'penalty': 'l2', 'solver': 'saga'})


rf_classifier_tfidf = RandomForestClassifier(**{'max_depth': 80, 'max_features': 'log2', 'n_estimators': 300})

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
        review_demoji = convert_demojis(review)
        punct_text    = remove_punctuation(review_demoji)
        contra_text   = expand_contractions(punct_text)
        review_toMs   = translate_text_deploy(contra_text,'en','ms')
        review_toEn   = translate_text_deploy(review_toMs,'ms','en')
        removed_review= remove_stopwords(review_toEn)
        token_text    = word_tokenize_wrapper(removed_review)
        nomalise_text = normalized_term(token_text)
        
        stemming_text = stemming(nomalise_text)
        lemma_text    = lemmatize_text(nomalise_text)

        # Vectorize the translated review
        review_vectorized = tfidf_vectorizer.transform([lemma_text])

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
