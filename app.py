from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

# Set page config as the first Streamlit command
st.set_page_config(page_title="GeminiDecode: Multilanguage Document Extraction by Gemini")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to process the uploaded image
def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Open the image using PIL
        image = Image.open(uploaded_file)
        return [image]  # Return a list containing the PIL Image object
    else:
        raise ValueError("No file uploaded")

# Function to get response from Gemini model
def get_gemini_response(input, image, prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content([input, image[0], prompt])
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

# Input prompt
input_prompt = """
You are an expert in understanding invoices.
We will upload an image as an invoice, and you will have to answer any questions based on the uploaded invoice image.
"""

# Streamlit UI
st.header("GeminiDecode: Multilanguage Document Extraction by Gemini")
text = ("Utilizing Gemini AI, this project effortlessly extracts vital information "
        "from diverse multilingual documents, transcending language barriers with precision and "
        "efficiency for enhanced productivity and decision-making.")
styled_text = f"<span style='font-family:serif;'>{text}</span>"
st.markdown(styled_text, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image of the document: ", type=["jpg", "jpeg", "png"])
image = ""
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

submit = st.button("Tell me about the document")

if submit:
    if uploaded_file is None:
        st.error("Please upload an image before submitting.")
    else:
        image_data = input_image_details(uploaded_file)
        response = get_gemini_response(input_prompt, image_data, input_prompt)
        st.subheader("The response is")
        st.write(response)