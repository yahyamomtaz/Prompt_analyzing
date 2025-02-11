from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
import matplotlib as plt

plt.use('TkAgg')

load_dotenv()

API_KEY = os.environ['OPENAI_API_KEY']
st.set_page_config(page_title="PandasAI Prompt Analyzer")
st.title('PandasAI Prompt Analyzer')

with st.sidebar:
    API_KEY = st.text_input("Enter your API KEY here:")
    
    st.markdown('''
    Made by **Yahya Momtaz**
    - [Visit my website](https://yayamomt.tech)
    - [GitHub](https://github.com/yahyamomtaz)
    - [Kaggle](https://www.kaggle.com/yahyamomtaz)
    - [Linkedin](https://www.linkedin.com/in/yahya-momtaz-601b34108)
    ''')

llm = OpenAI(api_token=API_KEY)

file = st.file_uploader("Upload your CSV file:", type=['csv'])

if file is not None:
    df = pd.read_csv(file)
    st.write(df.head())
    
    df_smart = SmartDataframe(df, config={"llm": llm})
    prompt = st.text_area("Enter your prompt here:")
    
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating, Please wait..."):
                st.write(df_smart.chat(prompt))
        else:
            st.write("Please Enter a Prompt")
        