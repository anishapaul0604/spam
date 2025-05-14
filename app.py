from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean and preprocess text for spam detection"""
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# Load the models
spam_detector = joblib.load("spam_detector.pkl")
vectorizer = spam_detector['vectorizer']
nb_model = spam_detector['nb_model']
lr_model = spam_detector['lr_model']
rf_model = spam_detector['rf_model']

def predict_spam(message):
    """Predict if a message is spam using multiple models"""
    cleaned_message = clean_text(message)
    vectorized_message = vectorizer.transform([cleaned_message])
    
    # Get predictions from all models
    nb_prediction = nb_model.predict(vectorized_message)
    lr_prediction = lr_model.predict(vectorized_message)
    rf_prediction = rf_model.predict(vectorized_message)
    
    # Get probabilities
    nb_probability = nb_model.predict_proba(vectorized_message)[0][1]
    lr_probability = lr_model.predict_proba(vectorized_message)[0][1]
    rf_probability = rf_model.predict_proba(vectorized_message)[0][1]

    return {
        "Naive Bayes": ("Spam" if nb_prediction[0] == 1 else "Not Spam", nb_probability),
        "Logistic Regression": ("Spam" if lr_prediction[0] == 1 else "Not Spam", lr_probability),
        "Random Forest": ("Spam" if rf_prediction[0] == 1 else "Not Spam", rf_probability)
    }

# Initialize Flask app
app = Flask(__name__)
    
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/spam_detection")
def spam_detection():
    return render_template("spam_detection.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    predictions = predict_spam(message)

    # Count how many models predicted Spam
    spam_count = sum(1 for label, _ in predictions.values() if label == "Spam")
    is_spam = spam_count >= 2  # Majority vote

    return render_template(
        "result.html",
        prediction=predictions,
        message=message,
        spam_count=spam_count,
        is_spam=is_spam
    )


if __name__ == '__main__':
    app.run(debug = True)