import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Sample dataset
data = {
    "question": [
        "What are the admission requirements?",
        "What courses are offered?",
        "What is the fee structure?",
        "Do you provide internships?",
        "What about placements?"
    ],
    "intent": ["admission", "courses", "fees", "internships", "placements"]
}

df = pd.DataFrame(data)

# Responses
responses = {
    "admission": "You need 12th pass with entrance exam.",
    "courses": "We offer B.Tech, MBA, B.Sc.",
    "fees": "Fees vary between 50k-1L per year.",
    "internships": "Yes, internships are provided.",
    "placements": "We have 90% placement record."
}

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])
y = df["intent"]

model = LogisticRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(responses, open("responses.pkl", "wb"))

print("Training complete!")
