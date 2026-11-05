
import re
import html

def clean_text(text: str) -> str:
    
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)       # HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # URLs
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text) # punctuation, keep apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text
