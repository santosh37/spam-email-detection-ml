import streamlit as st
import joblib

# Load saved model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Spam Email Detection")

st.title("Spam Email Detection System")
st.write("Enter email text below to check whether it is Spam or Not Spam.")

email_text = st.text_area("Enter Email Content")

if st.button("Check Email"):

    if email_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        vector = vectorizer.transform([email_text])
        prediction = model.predict(vector)
        score = model.decision_function(vector)

        risk_score = round(float(score[0]), 2)

        if prediction[0] == 1:
            st.error(f"This email is SPAM.\nRisk Score: {risk_score}")
        else:
            st.success(f"This email is NOT SPAM.\nConfidence Score: {risk_score}")
