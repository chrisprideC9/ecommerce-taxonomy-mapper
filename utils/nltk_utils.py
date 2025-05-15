import ssl
import nltk

def fix_nltk_ssl_issue():
    """Fix SSL certificate issue for NLTK downloads"""
    try:
        # Handle SSL certificate verification issue
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
            
        # Test if the fix worked by downloading a small dataset
        nltk.download('punkt', quiet=True)
        return True
    except Exception as e:
        print(f"Error fixing SSL for NLTK: {e}")
        return False