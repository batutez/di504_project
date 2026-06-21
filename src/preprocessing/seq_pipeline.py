"""
Section 6 — Shared GRU / ALBERT preprocessing pipeline.

"""
import re

URL_switch_pattern = r'https?://\S+|www\.\S+'
mention_switch_pattern = r'@\w+'
hashtag_switch_pattern = r'#\w+'
html_tag_removal_pattern = r'<.*?>'
html_entity_removal_pattern = r'&[a-zA-Z0-9#]+;'
special_char_removal_pattern =r'[^a-zA-Z0-9\s<>]' # this one keep letter spaces and < >
number_removal_pattern = r'\b\d+\b'

def clean_text_seq(text):
    """
    Clean a single text string for the GRU/ALBERT pipeline.

    Tasks accomplish in this function, in order; 
    1. lowercasing the text.
    2. removing html special tags and entities
    3. replacing URLs with the literal token '<URL>'.
    4. replacing @mentions with the literal token '<USER>'.
    5. replacing #hashtags with the literal token '<HASHTAG>'
    6. removing other special characters except < and >
    7. normalazing whitespace 

    Parameters
    ----------
    text : str

    Returns
    -------
    str
    """

   # lowercase the text
    text = text.lower()
    # remove html tags and entities
    text = re.sub(html_tag_removal_pattern,'',text)
    text = re.sub(html_entity_removal_pattern,'',text)
    # switch URL's with <URL>
    text = re.sub(URL_switch_pattern,'<URL>',text)
    # replace mentions with <USER>
    text = re.sub(mention_switch_pattern,'<USER>',text)
    # pleace #hashtags with <HASHTAG>
    text = re.sub(hashtag_switch_pattern,'<HASHTAG>',text)

    # remove special characters but keep < >
    text = re.sub(special_char_removal_pattern,'',text)
    # remove standalone numbers
    text = re.sub(number_removal_pattern,'',text)
    # normalize whitespace by collapsing multiple spaces, and striping from ends
    text = re.sub(r'\s+', ' ', text).strip()
    

    return text


if __name__ == "__main__": 
    test = """OMG!!! I am LOVING this new AI tool 😂😂 <br>

Visit: https://example.com/page?id=123 or www.testsite.com

Thanks @john_doe for sharing!!! #MachineLearning #AI2026

Tom &amp; Jerry were running and playing in 2026.

The price is 500 dollars!!! Python3 is amazing.

"""
    clenaed_text = clean_text_seq(test)
    print(f"test string : \n{test}")
    print("*"*100)
    print(f"result string : \n{clenaed_text}")