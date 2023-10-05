from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')

load_dotenv()

API_KEY = os.environ['OPENAI_API_KEY']
llm = OpenAI(api_token=API_KEY)
padndasai = PandasAI(llm)


st.title('PandasAI Prompt Analyzer')
file = st.file_uploader("Upload your CSV file", type=['csv'])

if file is not None:
    df = pd.read_csv(file)
    st.write(df.head())
    
    prompt = st.text_area("Enter your prompt here:")
    
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating, Please wait..."):
                st.write(padndasai.run(df,prompt=prompt))
        else:
            st.write("Please Enter a Prompt")