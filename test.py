def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

spam_detector = joblib.load("spam_detector.pkl")

vectorizer = spam_detector['vectorizer']
nb_model = spam_detector['nb_model']
lr_model = spam_detector['lr_model']
rf_model = spam_detector['rf_model']
clean_text = spam_detector['clean_text']

def predict_spam(message):
    cleaned_message = clean_text(message)
    vectorized_message = vectorizer.transform([cleaned_message])
    nb_prediction = nb_model.predict(vectorized_message)
    lr_prediction = lr_model.predict(vectorized_message)
    rf_prediction = rf_model.predict(vectorized_message)
    
    nb_probability = nb_model.predict_proba(vectorized_message)[0][1]
    lr_probability = lr_model.predict_proba(vectorized_message)[0][1]
    rf_probability = rf_model.predict_proba(vectorized_message)[0][1]

    return {
        "Naive Bayes": ("Spam" if nb_prediction[0] == 1 else "Not Spam", nb_probability),
        "Logistic Regression": ("Spam" if lr_prediction[0] == 1 else "Not Spam", lr_probability),
        "Random Forest": ("Spam" if rf_prediction[0] == 1 else "Not Spam", rf_probability)
    }

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]  # Ensure 'message' matches the form field name in your HTML
    predictions = predict_spam(message)
    
    return render_template("result.html", predictions=predictions, message=message)











# <!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Spam Detection Result</title>
#     <!-- Bootstrap CSS -->
#     <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
#     <!-- Animate.css for smooth effects -->
#     <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"/>
#     <style>
#         body {
#             background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
#             min-height: 100vh;
#             display: flex;
#             justify-content: center;
#             align-items: center;
#             font-family: 'Poppins', sans-serif;
#         }
#         .card {
#             border: none;
#             border-radius: 20px;
#             box-shadow: 0 8px 20px rgba(0,0,0,0.1);
#             animation: fadeInUp 1s;
#         }
#         .btn-back {
#             background-color: #6c63ff;
#             color: white;
#             border-radius: 50px;
#             padding: 10px 20px;
#             text-decoration: none;
#         }
#         .btn-back:hover {
#             background-color: #5a54e8;
#             color: white;
#         }
#     </style>
# </head>
# <body>

# <div class="container">
#     <div class="card p-5 text-center animate__animated animate__fadeInUp">
#         <h2 class="mb-4">🔍 Spam Detection Result</h2>
        
#         <div class="mb-4">
#             <h5>Analyzed Message:</h5>
#             <p class="text-muted">{{ message }}</p>
#         </div>

#         <div class="mb-4">
#             <h5>Predictions:</h5>
#             <div class="list-group">
#                 {% for model, result in predictions.items() %}
#                     <div class="list-group-item d-flex justify-content-between align-items-center">
#                         <strong>{{ model }}</strong>
#                         <span>
#                             {{ result[0] }} 
#                             (Confidence: {{ '%.2f'|format(result[1]*100) }}%)
#                         </span>
#                     </div>
#                 {% endfor %}
#             </div>
#         </div>

#         <a href="/" class="btn-back mt-3">🔙 Go Back</a>
#     </div>
# </div>

# <!-- Bootstrap JS (optional, for components) -->
# <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
# </body>
# </html>