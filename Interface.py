from streamrag import process_url, generate_answer
import streamlit as st
st.title("AI Interface for RAG")
URL1=st.sidebar.text_input("URL1")
URL2=st.sidebar.text_input("URL2")
URL3=st.sidebar.text_input("URL3")
process_button=st.sidebar.button("Process URL")
placeholder=st.empty()
if process_button:
    urls=[url for url in (URL1,URL2,URL3) if url !='']
    if len(urls)==0:
        placeholder.text("You must provide at least one url")
    else:
        process_url(urls)
query=st.text_input("Question")
if query:
    st.header("Answer")
    answer=generate_answer(query)
    st.write(answer)