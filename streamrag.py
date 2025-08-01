import numpy as np
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(np.__version__)
import langchain, langchain_community, langchain_groq
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
documents=[]
from langchain_community.document_loaders import UnstructuredURLLoader
import unstructured
from sentence_transformers import SentenceTransformer
import faiss as fvd
model = SentenceTransformer("all-MiniLM-L6-v2")
index=None
chunk_texts_list=None
from groq import Groq
os.environ["GROQ_API_KEY"] = api_key
groq_client = Groq()
def process_url(urls):
    global index
    global chunk_texts_list
    loader = UnstructuredURLLoader(urls=urls)
    print("Loading data scrapped from urls")
    docs = loader.load()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    textsplitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "], chunk_size=400, chunk_overlap=50)
    print("splitting documents into chunks")
    docs_chunks = textsplitter.split_documents(docs)
    chunk_texts_list = [doc.page_content for doc in docs_chunks]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunk_texts_list, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = fvd.IndexFlatIP(dimension)
    fvd.normalize_L2(embeddings)
    print("Adding documents to FAISS vector database")
    index.add(embeddings)
def generate_answer(query):
    query_embedding = model.encode([query], convert_to_numpy=True)
    fvd.normalize_L2(query_embedding)
    print("Print searching for similar chunks using semantic search")
    D, I = index.search(query_embedding, k=5)
    matched_texts = [chunk_texts_list[idx] for idx in I[0]]
    final_output_string = "\n".join(matched_texts)
    prompt = f'''Given the question and context below,generate the answer based on the context only. If you dont find the answer in the context say "I DONT KNOW". Strictly do not make things up.
    Question: {query}
    Context: {final_output_string}
    '''
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ],
        # max_tokens=2048
    )
    return response.choices[0].message.content
