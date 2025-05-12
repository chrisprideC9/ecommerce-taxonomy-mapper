import openai
import streamlit as st
from config.config import OPENAI_API_KEY

def init_openai():
    """Initialize OpenAI API with key"""
    if not OPENAI_API_KEY:
        st.warning("OpenAI API key not found. GPT-assisted classification will be disabled.")
        return False
    
    openai.api_key = OPENAI_API_KEY
    return True

def gpt_analyze_product(url, title, description):
    """Use GPT to extract potential taxonomy categories"""
    if not init_openai():
        return None
    
    # Prepare input text
    combined_text = f"URL: {url}\nTitle: {title}\nDescription: {description}"
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Extract the most likely product category path from the following information about an e-commerce product. Format as 'Category > Subcategory > Further subcategory'. Identify only the primary category path based on product type, not attributes. Be concise and specific."},
                {"role": "user", "content": combined_text}
            ],
            temperature=0.3
        )
        
        suggested_category = response.choices[0].message.content.strip()
        return suggested_category
    except Exception as e:
        st.error(f"Error with GPT API: {e}")
        return None

def enhance_low_confidence_matches(df, threshold=0.3, max_calls=10):
    """Use GPT to enhance matches that have low confidence scores"""
    if not init_openai():
        return df
    
    # Find low confidence rows
    low_conf_rows = df[df['confidence'] < threshold].head(max_calls)
    
    for idx, row in low_conf_rows.iterrows():
        url = row.get('Address', '')
        title = row.get('Title 1', '')
        description = row.get('Meta Description 1', '')
        
        # Get GPT suggestion
        suggested_category = gpt_analyze_product(url, title, description)
        
        if suggested_category:
            df.at[idx, 'gpt_suggested_category'] = suggested_category
            df.at[idx, 'needs_review'] = True
    
    return df