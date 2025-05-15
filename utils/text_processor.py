import re
import streamlit as st
import functools
import hashlib

# Cache for text processing operations
_TEXT_CACHE = {}

# Cache decorator
def cache_result(func):
    """Cache function results based on input arguments"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key from function name and arguments
        key_parts = [func.__name__]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key = hashlib.md5(str(key_parts).encode()).hexdigest()
        
        # Return cached result if available
        if key in _TEXT_CACHE:
            return _TEXT_CACHE[key]
        
        # Calculate and cache result
        result = func(*args, **kwargs)
        _TEXT_CACHE[key] = result
        return result
    
    return wrapper

# Try to import NLTK with fallback to simple processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    # Download required NLTK resources silently
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Fallback stopwords
FALLBACK_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
    'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
    'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
    'to', 'from', 'in', 'on', 'at', 'by', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
    'he', 'she', 'it', 'they', 'we', 'you', 'i', 'me', 'my', 'mine', 'your', 'yours',
    'his', 'her', 'hers', 'its', 'their', 'theirs', 'our', 'ours'
}

# Precompile regex patterns for better performance
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s]')
NUMBERS_PATTERN = re.compile(r'\d+')
WHITESPACE_PATTERN = re.compile(r'\s+')

@cache_result
def simple_tokenize(text):
    """Simple tokenization - cached for performance"""
    if not isinstance(text, str):
        return []
    
    # Replace special characters and compress whitespace
    text = SPECIAL_CHARS_PATTERN.sub(' ', text)
    text = WHITESPACE_PATTERN.sub(' ', text)
    
    # Split and return non-empty tokens
    return [t for t in text.strip().split() if t]

@cache_result
def simple_lemmatize(word):
    """Simple lemmatization for common patterns - cached for performance"""
    if not word or len(word) < 3:
        return word
        
    # Handle common suffixes
    if word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    if word.endswith('es') and len(word) > 3:
        return word[:-2]
    if word.endswith('ing') and len(word) > 5:
        return word[:-3]
    if word.endswith('ed') and len(word) > 4:
        return word[:-2]
    if word.endswith('ly') and len(word) > 4:
        return word[:-2]
    return word

@cache_result
def preprocess_text(text):
    """Clean and normalize text for better matching - with caching"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, special characters, numbers
    text = URL_PATTERN.sub('', text)
    text = SPECIAL_CHARS_PATTERN.sub(' ', text)
    text = NUMBERS_PATTERN.sub(' ', text)
    
    # Use NLTK if available
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
        except Exception:
            # Fallback to simple methods on error
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

@cache_result
def extract_url_segments(url):
    """Extract meaningful segments from URL path - with caching"""
    if not isinstance(url, str):
        return []
        
    try:
        # Remove protocol and domain
        path = re.sub(r'https?://[^/]+/', '', url)
        
        # Remove common non-descriptive segments
        path = re.sub(r'(index\.html|default\.aspx|\.php|\.html|\.js|\.json)', '', path)
        
        # Split path into segments
        segments = path.split('/')
        
        # Common segments to filter out
        common_segments = {
            'page', 'pages', 'products', 'product', 'category', 'categories', 
            'shop', 'store', 'blog', 'blogs', 'news', 'wpm', 'custom', 'web',
            'catalog', 'item', 'items', 'view', 'display', 'details', 'detail'
        }
        
        # Filter and process segments
        filtered_segments = []
        for segment in segments:
            # Skip empty or common segments
            if segment and segment.lower() not in common_segments:
                # Replace hyphens and underscores with spaces
                segment = segment.replace('-', ' ').replace('_', ' ')
                # Remove any remaining special characters
                segment = re.sub(r'[^\w\s]', '', segment)
                # Normalize whitespace
                segment = re.sub(r'\s+', ' ', segment).strip()
                if segment:
                    filtered_segments.append(segment)
        
        return filtered_segments
    except Exception:
        return []

def batch_preprocess_texts(texts):
    """Preprocess multiple texts at once for efficiency"""
    if not texts:
        return []
        
    # Use cache for each text to avoid redundant processing
    return [preprocess_text(text) for text in texts]

def clear_cache():
    """Clear the text processing cache"""
    global _TEXT_CACHE
    _TEXT_CACHE = {}
    return len(_TEXT_CACHE)