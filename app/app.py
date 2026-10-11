import streamlit as st
import joblib
import numpy as np

from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            color: #6366F1;
            font-size: 2.4rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            text-align: center;
            color: #9CA3AF;
            font-size: 1rem;
            margin-bottom: 1.8rem;
        }

        .result-box {
            padding: 1.4rem;
            border-radius: 15px;
            text-align: center;
            margin-top: 1.2rem;
            margin-bottom: 1.4rem;
        }

        .fake-result {
            background-color: #FEE2E2;
            border: 1px solid #FCA5A5;
            color: #991B1B;
        }

        .real-result {
            background-color: #DCFCE7;
            border: 1px solid #86EFAC;
            color: #166534;
        }

        .result-title {
            font-size: 1.6rem;
            font-weight: 700;
        }

        .result-description {
            font-size: 1rem;
            margin-top: 0.6rem;
        }

        .explanation-card {
            padding: 1rem;
            border-radius: 12px;
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-bottom: 1rem;
        }

        .footer {
            text-align: center;
            color: #9CA3AF;
            font-size: 0.85rem;
            margin-top: 2rem;
            padding-bottom: 1rem;
        }

        div.stButton > button {
            width: 100%;
            font-weight: 600;
            padding: 0.65rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "linear_svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer_compare.pkl"


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_model_and_vectorizer():

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    return model, vectorizer


# ============================================================
# EXPLAINABLE AI FUNCTION
# ============================================================

def explain_prediction(model, vectorizer, input_text, top_n=10):
    """
    Explains a Linear SVM prediction using TF-IDF feature
    values and the model's learned coefficients.

    Positive contribution supports Real News.
    Negative contribution supports Fake News.
    """

    # Convert input text into TF-IDF features
    text_vector = vectorizer.transform([input_text])

    # Get feature names
    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    # Get model coefficients
    coefficients = model.coef_[0]

    # Convert sparse vector into a normal array
    tfidf_values = text_vector.toarray()[0]

    # Calculate the contribution of every feature
    contributions = tfidf_values * coefficients

    # Find the largest positive contributions
    positive_indices = np.argsort(contributions)[::-1]

    # Find the largest negative contributions
    negative_indices = np.argsort(contributions)

    real_words = []
    fake_words = []

    # Collect words supporting Real News
    for index in positive_indices:

        if contributions[index] > 0:

            real_words.append(
                (
                    feature_names[index],
                    float(contributions[index])
                )
            )

        if len(real_words) >= top_n:
            break

    # Collect words supporting Fake News
    for index in negative_indices:

        if contributions[index] < 0:

            fake_words.append(
                (
                    feature_names[index],
                    float(abs(contributions[index]))
                )
            )

        if len(fake_words) >= top_n:
            break

    return real_words, fake_words


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📰 Fake News Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        An intelligent news-classification system using
        Natural Language Processing and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not MODEL_PATH.exists():

    st.error(
        f"Linear SVM model file was not found: {MODEL_PATH}"
    )

    st.stop()


if not VECTORIZER_PATH.exists():

    st.error(
        f"TF-IDF vectorizer file was not found: {VECTORIZER_PATH}"
    )

    st.stop()


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

try:

    model, vectorizer = load_model_and_vectorizer()

except Exception as error:

    st.error("The trained model could not be loaded.")
    st.exception(error)

    st.stop()


# ============================================================
# INTRODUCTION
# ============================================================

st.info(
    "Enter a complete news headline and article. The system "
    "classifies the text according to patterns learned from "
    "the training dataset."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Enter News Details")

title = st.text_input(
    "News Title",
    placeholder="Example: Government announces a new education policy"
)

text = st.text_area(
    "News Article",
    height=260,
    placeholder="Paste the complete news article here..."
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("🔍 Predict News", type="primary"):

    title = title.strip()
    text = text.strip()

    # --------------------------------------------------------
    # Validate title
    # --------------------------------------------------------

    if not title:

        st.warning("Please enter a news title.")

        st.stop()

    # --------------------------------------------------------
    # Validate article
    # --------------------------------------------------------

    if not text:

        st.warning("Please enter the news article text.")

        st.stop()

    # --------------------------------------------------------
    # Combine title and article
    # --------------------------------------------------------

    combined_text = f"{title} {text}"

    word_count = len(combined_text.split())

    # --------------------------------------------------------
    # Minimum text-length validation
    # --------------------------------------------------------

    if word_count < 20:

        st.warning(
            f"The entered article contains only {word_count} words. "
            "Please enter at least 20 words for a more meaningful prediction."
        )

        st.stop()

    # --------------------------------------------------------
    # Perform prediction
    # --------------------------------------------------------

    try:

        # Convert input text into TF-IDF features
        text_vector = vectorizer.transform([combined_text])

        # Predict class
        prediction = model.predict(text_vector)[0]

        # Get Linear SVM decision score
        decision_score = float(
            model.decision_function(text_vector)[0]
        )

        # Display-only confidence estimate
        confidence_estimate = min(
            99.9,
            50 + abs(decision_score) * 10
        )

        # Generate explanation
        real_words, fake_words = explain_prediction(
            model=model,
            vectorizer=vectorizer,
            input_text=combined_text,
            top_n=10
        )

        st.divider()

        st.subheader("Prediction Result")

        # ----------------------------------------------------
        # Display prediction result
        # ----------------------------------------------------

        if prediction == 0:

            st.markdown(
                """
                <div class="result-box fake-result">
                    <div class="result-title">
                        ⚠️ Potentially Fake News
                    </div>
                    <div class="result-description">
                        The article resembles fake-news examples
                        learned from the training dataset.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-box real-result">
                    <div class="result-title">
                        ✅ Potentially Real News
                    </div>
                    <div class="result-description">
                        The article resembles real-news examples
                        learned from the training dataset.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # Display metrics
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Confidence Estimate",
                f"{confidence_estimate:.2f}%"
            )

        with col2:

            st.metric(
                "Input Word Count",
                word_count
            )

        st.caption(
            "The confidence value is an estimate derived from "
            "the SVM decision score. It is not a calibrated probability."
        )

        # ----------------------------------------------------
        # Explainable AI section
        # ----------------------------------------------------

        st.divider()

        st.subheader("🔎 Prediction Explanation")

        st.write(
            "These words had the strongest influence on the model's "
            "classification for this input."
        )

        explanation_col1, explanation_col2 = st.columns(2)

        with explanation_col1:

            st.markdown("### 🟢 Real-News Indicators")

            if real_words:

                for word, contribution in real_words:

                    st.markdown(
                        f"""
                        <div class="explanation-card">
                            <strong>{word}</strong><br>
                            <small>
                                Positive contribution:
                                {contribution:.4f}
                            </small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.info(
                    "No strong real-news indicators were identified."
                )

        with explanation_col2:

            st.markdown("### 🔴 Fake-News Indicators")

            if fake_words:

                for word, contribution in fake_words:

                    st.markdown(
                        f"""
                        <div class="explanation-card">
                            <strong>{word}</strong><br>
                            <small>
                                Negative contribution:
                                {contribution:.4f}
                            </small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.info(
                    "No strong fake-news indicators were identified."
                )

        # ----------------------------------------------------
        # Detailed prediction information
        # ----------------------------------------------------

        st.divider()

        with st.expander("View Prediction Details"):

            predicted_class = (
                "Fake News"
                if prediction == 0
                else "Real News"
            )

            st.write(
                f"**Predicted class:** {predicted_class}"
            )

            st.write(
                f"**SVM decision score:** {decision_score:.4f}"
            )

            st.write(
                "**Feature representation:** TF-IDF"
            )

            st.write(
                "**Classification algorithm:** Linear SVM"
            )

            st.write(
                "**Training dataset:** ISOT Fake News Dataset"
            )

            st.write(
                f"**Number of extracted features:** "
                f"{text_vector.shape[1]}"
            )

        # ----------------------------------------------------
        # Disclaimer
        # ----------------------------------------------------

        st.warning(
            "This result is a machine-learning classification, "
            "not an independent fact-check. Always verify important "
            "claims using reliable sources and official information."
        )

    except Exception as error:

        st.error("An error occurred during prediction.")
        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        Developed using Python, Streamlit, TF-IDF, and Linear SVM.<br>
        Fake News Detection System — Academic Project
    </div>
    """,
    unsafe_allow_html=True
)