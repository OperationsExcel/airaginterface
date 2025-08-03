#Code for the Interface to be deployed on Sreamlit Cloud
import streamlit as st
st.set_page_config(page_title="Finesse Arogya", layout="wide")
import os
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
from streamrag import process_url, generate_answer, process_doc_patient,process_doc_research, process_doc_doctor
#st.title("Finesse Arogya: AI Interface for RAG System")
st.markdown("<h1 style='text-align: center;'>🧠 Finesse Arogya: AI-Powered RAG System</h1>", unsafe_allow_html=True)
st.markdown("---")
st.sidebar.header("🔗 URL Input")
URL1=st.sidebar.text_input("URL1")
URL2=st.sidebar.text_input("URL2")
URL3=st.sidebar.text_input("URL3")
#process_button=st.sidebar.button("Process URL")
process_button = st.sidebar.button("🚀 Process URLs")
placeholder=st.empty()
if process_button:
    urls=[url for url in (URL1,URL2,URL3) if url !='']
    if len(urls)==0:
        #placeholder.text("You must provide at least one url")
        placeholder.warning("⚠️ You must provide at least one URL.")
    else:
        with st.spinner("Processing URLs..."):
         process_url(urls)
        st.success("✅ URLs processed successfully!")

query=st.text_input("Question for data from websites")
if query:
    st.header("🌐 From Processed Websites")
    answer=generate_answer(query)
    st.write(answer)
patientquery=st.text_input("Question for Patient Experience")
if patientquery:
    st.header("🧑‍🤝‍🧑 Answer From Patient Experience")
    answer=process_doc_patient(patientquery)
    st.write(answer)
researchquery=st.text_input("Question for Research papers")
if researchquery:
    st.header("📚 Answer From Research Papers")
    answer=process_doc_research(researchquery)
    st.write(answer)
doctorquery=st.text_input("Question for Doctor Experience")
if doctorquery:
    st.header("👨‍⚕️ Answer From Doctor Experience")
    answer=process_doc_doctor(doctorquery)
    st.write(answer)