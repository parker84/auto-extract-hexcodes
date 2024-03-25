from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
from tqdm import tqdm
from decouple import config
import logging, coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', 'INFO'), logger=logger)


st.set_page_config(
    page_title='Auto Hexcode Extractor',
    page_icon='🗜️',
    initial_sidebar_state='collapsed'
)

st.title('🗜️ Auto Hexcode Extractor')
st.caption('Upload an image and automatically extract the dominant colors in the image as hex codes.')

with st.form(key='my_form'):
    uploaded_files = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    num_colors = st.slider('Number of colors to extract:', min_value=1, max_value=100, value=5)
    submit_button = st.form_submit_button(label='Extract Hex Codes', type='primary')

# ------------helpers
def extract_hex_codes(image_path, num_colors):
    # Load image
    img = Image.open(image_path)
    img = img.convert('RGB')
    
    # Resize image to reduce processing time (optional)
    # img = img.resize((100, 100))

    # Convert image to numpy array
    img_array = np.array(img)
    
    # Reshape array to 2D array (rows represent pixels)
    img_flat = img_array.reshape(-1, 3)
    
    # Use K-means clustering to find dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img_flat)
    
    # Get the RGB values of the cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # Convert RGB values to hexadecimal format
    hex_codes = ['#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b)) for r, g, b in cluster_centers]
    
    return hex_codes


def display_hexcodes(hex_codes):
    st.write('### Dominant colors:')
    n_cols = 5
    cols = st.columns(n_cols)
    i = 0
    for hex_code in hex_codes:
        with cols[i % n_cols]:
            # st.write(hex_code)
            st.write(f'<span style="color:{hex_code}; font-size: 20px;">{hex_code}</span>', unsafe_allow_html=True)
        i += 1
    with st.expander('Hex Codes', expanded=True):
        st.markdown(
            f"""
    ```py
    {hex_codes}
    ```
            """
        )

# ------------main
if submit_button and uploaded_files:
    for uploaded_file in tqdm(uploaded_files, desc='Processing images'):
        try:
            hex_codes = extract_hex_codes(uploaded_file, num_colors)
            display_hexcodes(hex_codes)
            st.image(uploaded_file, use_column_width=True)
        except Exception as e:
            logger.error(f'Error processing image: {uploaded_file.name}')
            logger.error(e)
            st.error(f'Error processing image: {uploaded_file.name}')