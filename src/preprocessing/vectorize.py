"""
Section 9 — TF-IDF vectorization for the ML baselines.
Fit on training data only, then transform val/test with the same vocabulary 
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from config import TFIDF_MAX_FEATURES

def fit_tfidf(train_texts, max_features=TFIDF_MAX_FEATURES , ngram_range=(1,2)):

    """
    Fit a TfidfVectorizer on the training text.

    Returns
    -------
    (TfidfVectorizer, scipy.sparse matrix)

    Verification:
        len(vectorizer.get_feature_names_out()) <= max_features.
        train_matrix.shape[0] == len(train_texts).
    """
    tfidf_vec = TfidfVectorizer(max_features=max_features,stop_words= 'english' , ngram_range=ngram_range)
    spmatrix = tfidf_vec.fit_transform(train_texts)
    ## for verification 
    print("number of features : {len(tfidf_vec.get_feature_names_out())}")
    print(f"sample from the features : {[f for f in tfidf_vec.get_feature_names_out() if ' ' in f][:10]}")
    return tfidf_vec, spmatrix

def transform_tfidf(vectorizer, texts):
    """
    Transform val/test text using an already-fitted vectorizer.

    """
    return vectorizer.transform(texts)
