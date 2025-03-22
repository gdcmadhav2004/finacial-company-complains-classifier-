import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import json 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# Load pre-trained models
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
tfidf_transformer = pickle.load(open("tfidf_transformer.pkl", "rb"))
model = pickle.load(open("logistic_regression.pkl", "rb"))

# Category mapping
topic_names = {
    0: "Bank account services",
    1: "Credit Card/Prepaid Card",
    2: "Mortgages/loans",
    3: "Others",
    4: "Theft/Dispute reporting"
}

# Function to predict complaint category
def predict_category(text):
    text_transformed = vectorizer.transform([text])
    text_tfidf = tfidf_transformer.transform(text_transformed)
    probabilities = model.predict_proba(text_tfidf)[0]
    predicted_index = np.argmax(probabilities)
    return topic_names[predicted_index], probabilities[predicted_index]

# Sidebar Navigation
page = st.sidebar.radio ("Navigate", [ "Classify Complaint", "Multi file  Processing", "Visualizations", "Feedback","Dataset Description"])


if page == "Classify Complaint":
    st.title("📝 Classify a Single Complaint")
    complaint_text = st.text_area("Enter your complaint below:", height=150)

    if st.button("Find Category"):
        if complaint_text.strip():
            category, confidence = predict_category(complaint_text)
            st.success(f"**Predicted Category:** {category} (Confidence: {confidence:.2f})")
        else:
            st.warning("Please enter a complaint before clicking the button.")
if page == "Multi file  Processing":
    st.title("📂 Upload a File for Batch Classification")
    uploaded_file = st.file_uploader("Upload a CSV file with complaints", type="csv")

    if uploaded_file:
        complaints_df = pd.read_csv(uploaded_file)
        complaints_df['Predicted Category'], complaints_df['Confidence'] = zip(
            *complaints_df['Complaint'].apply(lambda x: predict_category(x))
        )
        st.write("Classified Complaints:", complaints_df)


if page == "Visualizations":
    st.title("📊 Complaint Data Visualizations")
    try:
        # Load dataset
        with open("complaints-2021-05-14_08_16.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        df = pd.json_normalize(data)

        # Print column names to verify
        st.write("Available columns in dataset:", df.columns)

        # Use the correct column name
        complaint_column = [col for col in df.columns if "complaint" in col.lower()]
        if complaint_column:
            complaint_column = complaint_column[0]
        else:
            st.error("No column related to complaints found in the dataset.")
            st.stop()

        # Complaint Text Length Distribution
        st.subheader("Complaint Text Length Distribution")
        plt.figure(figsize=(8, 4))
        plot = df[complaint_column].astype(str).str.len()
        plt.hist(plot, bins=50)
        plt.xlabel("Character length of the Complaint")
        plt.ylabel("Number of Complaints")
        plt.title("Complaint Text Length Distribution")
        st.pyplot(plt)
        
        # Unigram Frequency Distribution
        st.subheader("Top 30 Unigrams in Complaints")
        def get_top_n_gram(corpus, n_gram_range, n=None):
            vec = CountVectorizer(ngram_range=(n_gram_range, n_gram_range), stop_words='english').fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            return words_freq[:n]
        
        unigram = get_top_n_gram(df[complaint_column], 1, 30)
        df_1 = pd.DataFrame(unigram, columns=["Unigram", "count"])

        plt.figure(figsize=(25, 15))
        figure = sns.barplot(x=df_1["Unigram"], y=df_1['count'])
        plt.title("Top 30 Unigrams in Complaints")
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Display Top 10 Unigrams as Table
        st.subheader("Top 10 Words in the Unigram Frequency")
        st.dataframe(df_1.head(10))
        
        # Classification Report and Confusion Matrix
        

        st.subheader("Confusion Matrix for Logistic Regression")
        image_path = "confu.png"  # Ensure the file is in the project directory
        st.image(image_path, caption="Confusion Matrix for Logistic Regression", use_column_width=True)
        
        st.subheader("Training Summary")
        image__path = "report.png"  # Ensure the file is in the project directory
        st.image(image__path, use_column_width=True)


    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'complaints.json' is in the project directory.")

        
if page == "Feedback":
    st.title("📝 User Feedback")
    feedback_text = st.text_area("Provide your feedback or report incorrect classification:")
    
    if st.button("Submit Feedback"):
        if feedback_text.strip():
            with open("feedback.txt", "a") as f:
                f.write(f"{feedback_text}\n")
            st.success("Thank you for your feedback!")
        else:
            st.warning("Please enter feedback before submitting.")
            

if page == "Dataset Description":
    st.title("📄 Dataset Description")
    try:
        with open("complaints-2021-05-14_08_16.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        df = pd.json_normalize(data)
        st.write("Displaying the first 100 rows of the dataset:")
        st.dataframe(df.head(100))
    except FileNotFoundError:
        st.error("Dataset file not found. Please make sure 'complaints.json' is in the project directory.")