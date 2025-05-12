import streamlit as st
import pandas as pd
import numpy as np
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.db_utils import get_db_connection, load_taxonomy_paths_with_vectors

def main():
    st.set_page_config(page_title="E-commerce Taxonomy Mapper", page_icon="🏷️", layout="wide")
    
    st.title("E-commerce Taxonomy Mapper")
    st.write("Upload your Screaming Frog data to map products to taxonomies")
    
    # Sidebar for settings
    st.sidebar.header("Matching Settings")
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1,
        step=0.01,
        help="Minimum similarity score required for a match"
    )
    
    # Store context setting (Approach 1)
    store_context = st.sidebar.text_input(
        "Store Context (optional)",
        placeholder="e.g., barber supplies, electronics, furniture",
        help="Provide context about the store type to improve matching"
    )
    
    store_context_weight = st.sidebar.slider(
        "Store Context Weight", 
        min_value=0.0, 
        max_value=5.0, 
        value=2.0,
        step=0.1,
        help="Weight given to store context in matching (higher = more influential)"
    )
    
    show_alternatives = st.sidebar.checkbox(
        "Show Alternative Matches", 
        value=True,
        help="Show alternative category matches for review"
    )
    
    batch_size = st.sidebar.number_input(
        "Batch Size",
        min_value=10,
        max_value=1000,
        value=100,
        help="Number of products to process in each batch"
    )
    
    # Database settings
    st.sidebar.header("Data Source")
    use_db = st.sidebar.checkbox("Use Database", value=True)
    use_sample_taxonomy = st.sidebar.checkbox("Use Sample Taxonomy", value=False)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Screaming Frog CSV", type=['csv'])
    
    if uploaded_file:
        # Load the CSV data
        df = pd.read_csv(uploaded_file)
        
        # Display data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Check if required columns are present
        required_columns = ['Address', 'Title 1', 'Meta Description 1']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return
        
        # Load taxonomy paths
        df_paths = load_taxonomy_paths_with_vectors(conn)
        
        if use_db:
            with st.spinner("Connecting to database..."):
                conn = get_db_connection()
                if conn:
                    st.info(f"Connected to database")
                    df_paths = load_taxonomy_paths_with_vectors(conn)
        
        if df_paths is None or df_paths.empty or use_sample_taxonomy:
            st.warning("Using sample taxonomy data")
            sample_file = "taxonomy_paths.csv"
            try:
                df_paths = pd.read_csv(sample_file)
            except Exception as e:
                st.warning(f"Error loading sample taxonomy file: {e}")
                sample_data = {
                    'id': range(1, 9),
                    'full_path': [
                        "Apparel & Accessories > Clothing",
                        "Apparel & Accessories > Clothing > Activewear",
                        "Apparel & Accessories > Clothing > Outerwear",
                        "Health & Beauty > Personal Care",
                        "Health & Beauty > Personal Care > Shaving & Grooming",
                        "Health & Beauty > Personal Care > Shaving & Grooming > Electric Razors",
                        "Health & Beauty > Personal Care > Shaving & Grooming > Shaving Brushes",
                        "Health & Beauty > Personal Care > Shaving & Grooming > Razors & Blades"
                    ]
                }
                df_paths = pd.DataFrame(sample_data)
        
        if df_paths is None or df_paths.empty:
            st.error("Could not load taxonomy paths from any source")
            return
        
        # Category filtering (Approach 2)
        st.sidebar.header("Category Filtering")
        top_level_categories = sorted(df_paths['full_path'].apply(lambda x: x.split('>')[0].strip()).unique())
        
        category_filter = st.sidebar.multiselect(
            "Filter to These Top-Level Categories",
            options=top_level_categories,
            default=[],
            help="Limit matching to these top-level categories"
        )
        
        # Apply category filter
        filtered_paths = df_paths
        if category_filter:
            filtered_paths = df_paths[df_paths['full_path'].apply(
                lambda x: any(x.startswith(cat) for cat in category_filter)
            )]
            if filtered_paths.empty:
                st.warning("No taxonomy paths match your category filter. Using all paths.")
                filtered_paths = df_paths
            else:
                st.success(f"Filtered to {len(filtered_paths)} taxonomy paths in selected categories")
        
        st.success(f"Loaded {len(df_paths)} taxonomy paths total")
        
        # Process button
        if st.button("Start Mapping"):
            # Process in batches
            progress_bar = st.progress(0)
            results = []
            
            total_batches = (len(df) + batch_size - 1) // batch_size
            
            start_time = time.time()
            
            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(df))
                batch = df.iloc[start_idx:end_idx].copy()
                
                # Process this batch
                batch_results = process_batch(
                    batch, 
                    filtered_paths,
                    store_context=store_context,
                    context_weight=store_context_weight,
                    threshold=similarity_threshold,
                    top_n=5 if show_alternatives else 1
                )
                
                results.append(batch_results)
                progress_bar.progress((i + 1) / total_batches)
            
            # Combine results
            result_df = pd.concat(results, ignore_index=True)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            st.success(f"Processing completed in {processing_time:.2f} seconds")
            
            # Show statistics
            matched_count = result_df['predicted_category'].notna().sum()
            total_count = len(result_df)
            match_rate = (matched_count / total_count) * 100 if total_count > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Products", total_count)
            col2.metric("Matched Products", matched_count)
            col3.metric("Match Rate", f"{match_rate:.1f}%")
            
            # Display the results
            st.subheader("Results")
            st.dataframe(result_df)
            
            # Download button
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Download Mapped Data",
                data=csv,
                file_name="taxonomy_mapped_products.csv",
                mime="text/csv"
            )

def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_url_segments(url):
    """Extract meaningful path segments from URL"""
    try:
        # Remove domain part
        path = re.sub(r'https?://[^/]+/', '', url)
        
        # Remove common non-descriptive segments
        path = re.sub(r'(index\.html|default\.aspx|\.php|\.html|\.js|\.json)', '', path)
        
        # Split path into segments
        segments = path.split('/')
        
        # Remove empty segments and common words
        filtered_segments = []
        common_segments = ['page', 'pages', 'products', 'product', 'category', 'categories', 
                          'shop', 'store', 'blog', 'blogs', 'news', 'wpm', 'custom', 'web']
        
        for segment in segments:
            if segment and segment.lower() not in common_segments:
                # Replace hyphens and underscores with spaces
                segment = segment.replace('-', ' ').replace('_', ' ')
                filtered_segments.append(segment)
        
        return ' '.join(filtered_segments)
    except Exception:
        return ""

def find_best_taxonomy_match_with_vectors(url, title, description, df_paths, 
                                          store_context=None, context_weight=2.0, 
                                          top_n=5, threshold=0.0):
    """Find the best matching taxonomy path using pre-computed vectors"""
    from utils.vector_utils import vectorize_text
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Extract meaningful parts from URL
    url_segments = extract_url_segments(url)
    
    # Combine all text data
    combined_text = f"{' '.join(url_segments)} {title} {title} {description}"
    
    # Add store context if provided
    if store_context and store_context.strip():
        context_repeat = max(1, int(context_weight))
        context_text = ' '.join([store_context] * context_repeat)
        combined_text = f"{combined_text} {context_text}"
    
    cleaned_text = clean_text(combined_text)
    
    # Create vector for the product text
    product_vector = vectorize_text(cleaned_text)
    
    # Convert stored vectors from string to numpy arrays if needed
    if isinstance(df_paths['vector'].iloc[0], str):
        from utils.vector_utils import prepare_vectors
        df_paths = prepare_vectors(df_paths, 'vector')
        vector_column = 'content_vector'
    else:
        vector_column = 'vector'
    
    # Calculate similarities
    similarities = []
    for idx, row in df_paths.iterrows():
        category_vector = row[vector_column]
        if category_vector is not None:
            # Handle vector format
            if isinstance(category_vector, str):
                # Parse string to vector if needed
                import ast
                category_vector = np.array(ast.literal_eval(category_vector))
            
            # Calculate cosine similarity
            similarity = cosine_similarity([product_vector], [category_vector])[0][0]
            similarities.append(similarity)
        else:
            similarities.append(0)
    
    # Add similarities to dataframe
    result_df = df_paths.copy()
    result_df['similarity'] = similarities
    
    # Sort by similarity
    result_df = result_df.sort_values('similarity', ascending=False)
    
    # Filter by threshold
    if threshold > 0:
        result_df = result_df[result_df['similarity'] >= threshold]
    
    return result_df.head(top_n)

def process_batch(batch, df_paths, store_context=None, context_weight=2.0, 
                 threshold=0.1, top_n=5):
    """Process a batch of products to map to taxonomy paths"""
    for idx, row in batch.iterrows():
        try:
            url = row['Address']
            title = row['Title 1']
            description = row.get('Meta Description 1', '')
            
            # Find matches
            matches = find_best_taxonomy_match_with_vectors(
                url=url,
                title=title,
                description=description,
                taxonomy_paths_df=df_paths,
                store_context=store_context,
                context_weight=context_weight,
                top_n=top_n,
                threshold=0.0  # Get all matches, filter later
            )
            
            if not matches.empty:
                top_match = matches.iloc[0]
                
                # Only assign if above threshold
                if top_match['similarity'] >= threshold:
                    batch.at[idx, 'predicted_category'] = top_match['full_path']
                    batch.at[idx, 'category_id'] = top_match['id']
                    batch.at[idx, 'confidence'] = float(top_match['similarity'])
                else:
                    batch.at[idx, 'predicted_category'] = None
                    batch.at[idx, 'category_id'] = None
                    batch.at[idx, 'confidence'] = float(top_match['similarity'])
                
                # Add top matches for review
                match_details = []
                for i, match in matches.iterrows():
                    match_details.append(f"{match['full_path']} ({match['similarity']:.3f})")
                batch.at[idx, 'alternative_matches'] = "; ".join(match_details[:5])
            else:
                batch.at[idx, 'predicted_category'] = None
                batch.at[idx, 'category_id'] = None
                batch.at[idx, 'confidence'] = 0.0
                
        except Exception as e:
            st.error(f"Error processing row {idx}: {e}")
            batch.at[idx, 'predicted_category'] = None
            batch.at[idx, 'category_id'] = None
            batch.at[idx, 'confidence'] = 0.0
            batch.at[idx, 'error'] = str(e)
    
    return batch

if __name__ == "__main__":
    main()