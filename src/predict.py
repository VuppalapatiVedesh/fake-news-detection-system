import joblib

# Load the trained Linear SVM model and TF-IDF vectorizer
model = joblib.load("models/linear_svm_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer_compare.pkl")


def predict_news(title, text):
    """
    Predict whether a news article resembles fake or real news.
    """

    title = title.strip()
    text = text.strip()

    combined_text = title + " " + text

    word_count = len(combined_text.split())

    if word_count < 20:
        return {
            "prediction": "Insufficient Text",
            "message": "Please enter a more detailed news article."
        }

    text_vector = vectorizer.transform([combined_text])

    prediction = model.predict(text_vector)[0]

    if prediction == 0:
        result = "Fake News"
    else:
        result = "Real News"

    return {
        "prediction": result,
        "message": (
            "This is a machine-learning prediction based on textual "
            "patterns. It is not an independent fact-check."
        )
    }


if __name__ == "__main__":

    title = input("Enter news title: ")
    text = input("Enter news text: ")

    result = predict_news(title, text)

    print("\nPrediction:", result["prediction"])
    print("Note:", result["message"])