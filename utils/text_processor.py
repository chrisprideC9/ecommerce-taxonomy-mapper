import re
import streamlit as st

# Try to import NLTK resources, but have fallbacks if they're not available
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Fix SSL certificate issue for NLTK downloads
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    # Download required NLTK resources
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    st.warning("NLTK not available, using simplified text processing")

# English stopwords as fallback if NLTK not available
FALLBACK_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
    'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
    'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
    'to', 'from', 'in', 'on', 'at', 'by', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
    'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'my', 'mine', 'your', 'yours',
    'his', 'her', 'hers', 'its', 'their', 'theirs', 'our', 'ours'
}

def simple_tokenize(text):
    """Simple tokenization function that splits text on whitespace and punctuation"""
    # Replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Split on whitespace
    return text.strip().split()

def simple_lemmatize(word):
    """Simple lemmatization for common English words"""
    # Basic rules for common English patterns
    if word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    if word.endswith('es'):
        return word[:-2]
    if word.endswith('ing'):
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    return word

def preprocess_text(text):
    """Clean and normalize text for better matching"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Use NLTK if available, otherwise use simple methods
    if NLTK_AVAILABLE:
        try:
            # Tokenize
            tokens = nltk.word_tokenize(text)
            
            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
            
            # Lemmatize
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        except Exception as e:
            st.warning(f"NLTK processing error: {e}. Using fallback method.")
            # Use simpler methods as fallback
            tokens = simple_tokenize(text)
            tokens = [word for word in tokens if word not in FALLBACK_STOPWORDS and len(word) > 1]
            tokens = [simple_lemmatize(word) for word in tokens]
    else:
        # Use simple methods
        tokens = simple_tokenize(text)
        tokens = [word for word in tokens if word not in FALLBACK_STOPWORDS and len(word) > 1]
        tokens = [simple_lemmatize(word) for word in tokens]
    
    # Join tokens back into a string
    return ' '.join(tokens)

def extract_url_segments(url):
    """Extract meaningful segments from URL path"""
    try:
        # Remove protocol and domain
        path = re.sub(r'https?://[^/]+/', '', url)
        
        # Remove common segments that don't indicate category
        path = re.sub(r'(index\.html|default\.aspx|\.php|\.html|\.js|\.json)', '', path)
        
        # Split path into segments
        segments = path.split('/')
        
        # Remove empty segments and common non-category segments
        filtered_segments = []
        common_segments = ['page', 'pages', 'products', 'product', 'category', 'categories', 
                          'shop', 'store', 'blog', 'blogs', 'news', 'wpm', 'custom', 'web']
        
        for segment in segments:
            # Skip empty segments and common non-descriptive segments
            if segment and segment.lower() not in common_segments:
                # Replace hyphens and underscores with spaces
                segment = segment.replace('-', ' ').replace('_', ' ')
                filtered_segments.append(segment)
        
        return filtered_segments
    except Exception:
        return []

def combine_text_features(url, title, description=None, weights=(1, 2, 1)):
    """Combine different text features with weights for importance"""
    url_weight, title_weight, desc_weight = weights
    
    # Extract URL segments
    url_segments = extract_url_segments(url)
    url_text = ' '.join(url_segments) * url_weight
    
    # Weight the title
    title_text = (title or '') * title_weight
    
    # Add description if available
    desc_text = (description or '') * desc_weight if description else ''
    
    # Combine all texts
    combined_text = f"{url_text} {title_text} {desc_text}"
    
    return combined_text