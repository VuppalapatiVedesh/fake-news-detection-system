
from pathlib import Path

import html
import joblib
import numpy as np
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "linear_svm_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer_compare.pkl"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            color: #6366F1;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            text-align: center;
            color: #9CA3AF;
            margin-bottom: 1.5rem;
        }

        .result-box {
            padding: 1.3rem;
            border-radius: 14px;
            text-align: center;
            margin: 1rem 0;
        }

        .fake-result {
            background: #FEE2E2;
            border: 1px solid #FCA5A5;
            color: #991B1B;
        }

        .real-result {
            background: #DCFCE7;
            border: 1px solid #86EFAC;
            color: #166534;
        }

        .result-title {
            font-size: 1.4rem;
            font-weight: 700;
        }

        .explanation-card {
            padding: 0.8rem;
            border-radius: 10px;
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-bottom: 0.6rem;
        }

        .footer {
            text-align: center;
            color: #9CA3AF;
            font-size: 0.8rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def get_class_name(class_value):
    """
    The project's expected label mapping:
        0 = Fake News
        1 = Real News
    """
    if class_value == 0:
        return "Fake News"

    if class_value == 1:
        return "Real News"

    return str(class_value)


# ============================================================
# EXPLAIN LINEAR SVM PREDICTIONS
# ============================================================

def explain_prediction(model, vectorizer, input_text, top_n=10):
    """
    For binary Linear SVM:
    Positive coefficient contributions support model.classes_[1].
    Negative coefficient contributions support model.classes_[0].
    """

    text_vector = vectorizer.transform([input_text])

    if not hasattr(model, "coef_"):
        return [], [], model.classes_[0], model.classes_[-1]

    if len(model.classes_) != 2 or model.coef_.shape[0] != 1:
        return [], [], model.classes_[0], model.classes_[-1]

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    # Sparse matrix multiplication avoids converting all 50,000
    # features into a dense array.
    contributions = np.asarray(
        text_vector.multiply(coefficients).toarray()
    ).ravel()

    present_indices = text_vector.indices

    positive_indices = sorted(
        (
            i for i in present_indices
            if contributions[i] > 0
        ),
        key=lambda i: contributions[i],
        reverse=True,
    )

    negative_indices = sorted(
        (
            i for i in present_indices
            if contributions[i] < 0
        ),
        key=lambda i: contributions[i],
    )

    positive_words = [
        (feature_names[i], float(contributions[i]))
        for i in positive_indices[:top_n]
    ]

    negative_words = [
        (feature_names[i], float(contributions[i]))
        for i in negative_indices[:top_n]
    ]

    return (
        positive_words,
        negative_words,
        model.classes_[1],
        model.classes_[0],
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📰 Fake News Detection System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        News-text classification using Natural Language Processing
        and Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Enter a news headline and article. The system predicts a class "
    "using patterns learned from its training dataset. It does not "
    "independently verify news claims."
)


# ============================================================
# VERIFY AND LOAD ARTIFACTS
# ============================================================

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not VECTORIZER_PATH.exists():
    st.error(f"TF-IDF vectorizer not found: {VECTORIZER_PATH}")
    st.stop()

try:
    model, vectorizer = load_model_and_vectorizer()

except Exception as error:
    st.error("Could not load the saved model or vectorizer.")
    st.exception(error)
    st.stop()


# Show the saved model's actual class order.
with st.expander("Model information"):
    st.write("Classifier:", type(model).__name__)
    st.write("Model classes:", model.classes_.tolist())
    st.write("TF-IDF features:", len(vectorizer.get_feature_names_out()))

    if len(model.classes_) == 2:
        st.caption(
            f"Negative decision scores favor "
            f"{get_class_name(model.classes_[0])}; positive scores favor "
            f"{get_class_name(model.classes_[1])}."
        )


# ============================================================
# INPUT FORM
# ============================================================

st.subheader("Enter News Details")

with st.form("news_prediction_form"):
    title = st.text_input(
        "News Title",
        placeholder="Enter the news headline",
    )

    article_text = st.text_area(
        "News Article",
        height=240,
        placeholder="Paste the complete news article here",
    )

    submitted = st.form_submit_button(
        "🔍 Predict News",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:
    title = title.strip()
    article_text = article_text.strip()

    if not title:
        st.warning("Please enter a news headline.")

    elif not article_text:
        st.warning("Please enter the news article text.")

    else:
        combined_text = f"{title} {article_text}"
        word_count = len(combined_text.split())

        if word_count < 20:
            st.warning(
                "Please provide at least 20 words for this test. "
                "Longer text is not necessarily more accurate."
            )

        elif len(model.classes_) != 2:
            st.error(
                "This application expects a binary classification model."
            )

        else:
            try:
                text_vector = vectorizer.transform([combined_text])

                # Use the model's own prediction.
                prediction = model.predict(text_vector)[0]

                # For a binary Linear SVM, a positive decision score
                # favors classes_[1], and a negative score favors classes_[0].
                decision_score = float(
                    np.asarray(
                        model.decision_function(text_vector)
                    ).ravel()[0]
                )

                predicted_name = get_class_name(prediction)

                st.divider()
                st.subheader("Prediction Result")

                if prediction == 0:
                    st.markdown(
                        """
                        <div class="result-box fake-result">
                            <div class="result-title">
                                ⚠️ Potentially Fake News
                            </div>
                            <p>
                                The model classified this text as Fake News.
                                This is not proof that the article is false.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                elif prediction == 1:
                    st.markdown(
                        """
                        <div class="result-box real-result">
                            <div class="result-title">
                                ✅ Potentially Real News
                            </div>
                            <p>
                                The model classified this text as Real News.
                                This is not independent verification of its claims.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                else:
                    st.warning(f"Predicted class: {predicted_name}")

                # Do not call the decision score a probability.
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "SVM Decision Score",
                        f"{decision_score:.4f}",
                    )

                with col2:
                    st.metric("Input Word Count", word_count)

                st.caption(
                    "The SVM decision score is not a probability or "
                    "a calibrated confidence percentage. Its sign and "
                    "magnitude are meaningful relative to this model's "
                    "decision boundary."
                )

                # ------------------------------------------------
                # EXPLANATION
                # ------------------------------------------------

                st.divider()
                st.subheader("🔎 Prediction Explanation")

                (
                    positive_words,
                    negative_words,
                    positive_class,
                    negative_class,
                ) = explain_prediction(
                    model,
                    vectorizer,
                    combined_text,
                    top_n=10,
                )

                st.write(
                    f"**Positive contributions** favor "
                    f"**{get_class_name(positive_class)}**."
                )
                st.write(
                    f"**Negative contributions** favor "
                    f"**{get_class_name(negative_class)}**."
                )

                left_col, right_col = st.columns(2)

                with left_col:
                    st.markdown(
                        f"#### 🟢 Supports {get_class_name(positive_class)}"
                    )

                    if positive_words:
                        for word, contribution in positive_words:
                            safe_word = html.escape(word)

                            st.markdown(
                                f"""
                                <div class="explanation-card">
                                    <strong>{safe_word}</strong><br>
                                    Contribution: {contribution:+.4f}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No positive feature contributions found.")

                with right_col:
                    st.markdown(
                        f"#### 🔴 Supports {get_class_name(negative_class)}"
                    )

                    if negative_words:
                        for word, contribution in negative_words:
                            safe_word = html.escape(word)

                            st.markdown(
                                f"""
                                <div class="explanation-card">
                                    <strong>{safe_word}</strong><br>
                                    Contribution: {contribution:+.4f}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.info("No negative feature contributions found.")

                # ------------------------------------------------
                # DETAILS
                # ------------------------------------------------

                with st.expander("View Prediction Details"):
                    st.write("**Predicted class:**", predicted_name)
                    st.write("**Model classes:**", model.classes_.tolist())
                    st.write("**SVM decision score:**", f"{decision_score:.6f}")
                    st.write("**Feature representation:** TF-IDF")
                    st.write("**Classifier:** Linear SVM")
                    st.write("**Extracted feature count:**", text_vector.shape[1])
                    st.write("**Training dataset:** ISOT Fake News Dataset")

                st.warning(
                    "This system classifies text using learned patterns. "
                    "It does not independently fact-check claims. Verify "
                    "important news using reliable, independent sources."
                )

            except Exception as error:
                st.error("An error occurred while predicting the article.")
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
    unsafe_allow_html=True,
)
