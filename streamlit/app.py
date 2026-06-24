import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / ".env")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

st.set_page_config(
    page_title="Blood Report Analyzer",
    page_icon="🩸",
    layout="wide"
)

st.title("🩸 Blood Report Analyzer")
st.write(
    "Upload a blood report (.txt file) and get blood parameter analysis, health summary, and diet recommendations."
)

uploaded_file = st.file_uploader(
    "Upload Blood Report",
    type=["txt"]
)

if uploaded_file:

    blood_report = uploaded_file.read().decode("utf-8")

    with st.expander("View Blood Report"):
        st.text_area(
            "Report Content",
            blood_report,
            height=300
        )

    if st.button("Analyze Report"):

        with st.spinner("Analyzing blood report..."):

            extraction_prompt = f"""
            You are a medical data extraction assistant.

            From the blood report below, extract all test values and classify each one as HIGH, LOW or NORMAL
            based on the reference ranges provided in the report.

            Format your response as:
            - Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

            Blood Report:
            {blood_report}
            """

            extraction_response = llm.invoke(extraction_prompt)

            extracted_values = (
                extraction_response.content
                if hasattr(extraction_response, "content")
                else str(extraction_response)
            )

            diet_prompt = f"""
            You are a clinical nutritionist specializing in Indian dietary habits.

            Based on the blood work analysis below, write:

            1. A short health summary in 2 lines explaining the patient's condition in simple language.

            2. A short, practical Indian diet plan having only two sections:
               (1) Foods to avoid
               (2) Foods to eat more of

            Do not include any other sections.

            Blood Work Analysis:
            {extracted_values}
            """

            diet_response = llm.invoke(diet_prompt)

            diet_plan = (
                diet_response.content
                if hasattr(diet_response, "content")
                else str(diet_response)
            )

        st.success("Analysis Complete!")

        tab1, tab2 = st.tabs(
            ["📊 Blood Analysis", "🥗 Diet Recommendation"]
        )

        with tab1:
            st.markdown(extracted_values)

        with tab2:
            st.markdown(diet_plan)