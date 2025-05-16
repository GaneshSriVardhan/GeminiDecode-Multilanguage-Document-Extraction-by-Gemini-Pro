from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import PyPDF2
import pandas as pd
import json
import docx
import io

# Set page config with better design
st.set_page_config(page_title="GeminiDecode: Multilingual Document Q&A", layout="wide")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Custom styles to make the UI look better
st.markdown("""
    <style>
        body {
            font-family: 'Helvetica', sans-serif;
            background-color: #f4f7f6;
            color: #333;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 5px;
        }
        .stTextArea textarea {
            background-color: #ffffff;
            color: #333;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .stMarkdown {
            font-size: 18px;
            font-weight: 400;
        }
        .stFileUploader {
            margin-top: 20px;
        }
        .stImage {
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to process different file types
def process_file(uploaded_file, file_type):
    try:
        if file_type in ["jpg", "jpeg", "png"]:
            image = Image.open(uploaded_file)
            return "", [image]  # Return empty text and image for Gemini to process
        elif file_type == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text, None
        elif file_type == "txt":
            text = uploaded_file.read().decode("utf-8")
            return text, None
        elif file_type in ["csv", "xlsx", "xls"]:
            if file_type == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            text = df.to_string()
            return text, None
        elif file_type == "json":
            data = json.load(uploaded_file)
            text = json.dumps(data, indent=2)
            return text, None
        elif file_type == "docx":
            doc = docx.Document(uploaded_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text, None
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")

# Function to get response from Gemini model
def get_gemini_response(input_text, image, user_question):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Base prompt to ensure multilingual handling and English output
        base_prompt = """
        You are an expert in understanding documents in any language. 
        The user has provided a document and a question about it. 
        Analyze the document, identify its language if non-English, and answer the user's question in English. 
        If the question involves extracting specific information (e.g., dates, names, amounts, headings, pages), provide the extracted details. 
        If the question asks for a summary or key points, provide a concise response (100 words or less). 
        If the document is irrelevant to the question, state so clearly.
        """
        full_prompt = f"{base_prompt}\n\nUser Question: {user_question}"
        
        if image:
            response = model.generate_content([full_prompt, input_text, image[0]])
        else:
            response = model.generate_content([full_prompt, input_text])
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"

# Streamlit UI
st.title("GeminiDecode: Multilingual Document Q&A by Gemini")
st.markdown("""
    <h3 style='font-weight:400;'>Upload documents in any language and ask questions about them. Gemini AI will provide answers in English, transcending language barriers.</h3>
""", unsafe_allow_html=True)

# File uploader for multiple formats
uploaded_file = st.file_uploader(
    "Choose a document: ",
    type=["jpg", "jpeg", "png", "pdf", "txt", "csv", "json", "xlsx", "xls", "docx"]
)

# Text input for user question
user_question = st.text_area(
    "Ask any question about the document:",
    placeholder="e.g., What is the total amount on the invoice? Summarize the content under 'Introduction'. What are the key points on page 3?",
    height=150
)

# Display uploaded file
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()
    if file_type in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    else:
        st.write(f"Uploaded file: {uploaded_file.name}")

submit = st.button("Get Answer")

if submit:
    if uploaded_file is None:
        st.error("Please upload a document before submitting.")
    elif not user_question.strip():
        st.error("Please enter a question about the document.")
    else:
        try:
            # Process the uploaded file
            extracted_text, image_data = process_file(uploaded_file, file_type)
            
            # Get response from Gemini based on user question
            response = get_gemini_response(extracted_text, image_data, user_question)
            
            # Display results
            st.subheader("Answer")
            st.write(response)
        except ValueError as e:
            st.error(str(e))
