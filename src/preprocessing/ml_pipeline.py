"""
Section 6 — ML TEXT preprocessing pipeline.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

html_removal_pattern = r'https?://\S+|www\.\S+'
mention_removal_pattern = r'@\w+'
hashtag_removal_pattern = r'#\w+'
html_tag_removal_pattern = r'<.*?>'
html_entity_removal_pattern = r'&[a-zA-Z0-9#]+;'
special_char_removal_pattern = r'[^a-zA-Z\s]' # this one keep only letters and spaces
number_removal_pattern = r'\b\d+\b'

def clean_text_ml(text):
    """
    Clean a single text string for the ML pipeline, that uses TF-IDF as feature .

    1. lowercasing the text.
    2. removing URLs 
    3. removing @mentions.
    4. removing #hashtags (the whole token: '#' and the word).
    5. removing HTML tags/entities (e.g. <br>, &amp;).
    6. removing special characters / punctuation (keep only letters and spaces).
    7. removing numbers.
    8. normalizing for whitespace and strip
    9. tokenize with word_tokenizer
    10. removing stopwords with NLTK stop_word for English language
    11. lemmatizing each token with WordNetLemmatizer, position set to 'v' for verbs

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Preprocessed, cleaned text that is ready for TF-IDF.

    """
    # lowercase the text
    text = text.lower()
    # remove HTML and ww patterns
    text = re.sub(html_removal_pattern,'',text)
    # remove @mentions
    text = re.sub(mention_removal_pattern,'',text)
    # remove #hashtags
    text = re.sub(hashtag_removal_pattern,'',text)
    # remove html tags and entities
    text = re.sub(html_tag_removal_pattern,'',text)
    text = re.sub(html_entity_removal_pattern,'',text)
    # remove special characters
    text = re.sub(special_char_removal_pattern,'',text)
    # remove standalone numbers
    text = re.sub(number_removal_pattern,'',text)
    # normalize whitespace by collapsing multiple spaces, and striping from ends
    text = re.sub(r'\s+', ' ', text).strip()
    # tokenize
    tokens = word_tokenize(text)
    # stopword removal
    tokens_filtered = [token for token in tokens if token not in stop_words]
    # lemmatize 
    tokens_lemmatized = [lemmatizer.lemmatize(token, pos='v') for token in tokens_filtered]
    # reunify under single text string
    text_reunified = " ".join(tokens_lemmatized)

    return text_reunified


if __name__ == "__main__": 
    test = """OMG!!! I am LOVING this new AI tool 😂😂 <br>

Visit: https://example.com/page?id=123 or www.testsite.com

Thanks @john_doe for sharing!!! #MachineLearning #AI2026

Tom &amp; Jerry were running and playing in 2026.

The price is 500 dollars!!! Python3 is amazing.

"""
    clenaed_text = clean_text_ml(test)
    print(f"test string : \n{test}")
    print("*"*100)
    print(f"result string : \n{clenaed_text}")