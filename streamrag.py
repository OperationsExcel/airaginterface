import numpy as np
from dotenv import load_dotenv
import os

if "GROQ_API_KEY" not in os.environ:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
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

groq_client = Groq()
def process_url(urls):
    global index
    global chunk_texts_list
    loader = UnstructuredURLLoader(urls=urls)
    print("Loading data scrapped from urls")
    docs = loader.load()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    textsplitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "], chunk_size=250, chunk_overlap=30)
    print("splitting documents into chunks")
    docs_chunks = textsplitter.split_documents(docs)
    chunk_texts_list = [doc.page_content for doc in docs_chunks]
    embeddings = model.encode(chunk_texts_list, convert_to_numpy=True)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    embeddings = embeddings.astype(np.float32)
    dimension = embeddings.shape[1]
    index = fvd.IndexFlatIP(dimension)
    fvd.normalize_L2(embeddings)
    print("Adding documents to FAISS vector database")
    index.add(embeddings)
def generate_answer(query):
    global index, chunk_texts_list

    if index is None or chunk_texts_list is None or len(chunk_texts_list) == 0:
        return "FAISS index is empty or chunk data is missing"
    query_embedding = model.encode([query], convert_to_numpy=True)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = query_embedding.astype(np.float32)
    fvd.normalize_L2(query_embedding)
    print("Print searching for similar chunks using semantic search")
    #dimension = query_embedding.shape[1]
    #index = fvd.IndexFlatIP(dimension)
    #print(f"Index created with dimension: {dimension}")
    D, I = index.search(query_embedding, k=5)
    if I is None or len(I[0]) == 0:
        return "No results found in the vector DB."
    matched_texts = [chunk_texts_list[idx] for idx in I[0] if 0 <= idx < len(chunk_texts_list)]
    final_output_string = "\n".join(matched_texts)
    print(final_output_string)
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
