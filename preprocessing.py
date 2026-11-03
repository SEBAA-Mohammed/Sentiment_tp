"""
preprocessing.py
Text cleaning shared by train.py AND app.py, so that training and prediction
use exactly the same preprocessing (very important!).
"""
import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
APOSTROPHE_RE = re.compile(r"[`´’‘]")   # this dataset writes "don`t" with a backtick
REPEAT_RE = re.compile(r"(.)\1{2,}")     # sooooo -> soo


def clean_text(text) -> str:
    """Lowercase, remove URLs / @mentions / punctuation / digits, collapse spaces.

    Notes:
    - We do NOT remove stop-words: words like "not", "no", "never" carry
      sentiment ("not good" != "good").
    - Backticks/curly quotes are turned into normal apostrophes so that
      "don`t" becomes "don't" (instead of being split into "don" + "t").
    - Accented letters (a-z + à-ÿ) are kept so French text also works.
    """
    text = html.unescape(str(text)).lower()   # &amp; -> &, lowercase
    text = APOSTROPHE_RE.sub("'", text)       # don`t -> don't
    text = URL_RE.sub(" ", text)              # remove links
    text = MENTION_RE.sub(" ", text)          # remove @user
    text = text.replace("#", " ")             # keep hashtag word, drop '#'
    text = REPEAT_RE.sub(r"\1\1", text)       # yayyyyy -> yayy
    text = re.sub(r"[^a-zà-ÿ\s']", " ", text) # keep letters and apostrophes
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text
