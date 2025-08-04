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
patient_index=None
research_index=None
chunk_texts_list=None
patient_text_chunks=None
research_text_chunks=None
global doctor_index
global doctor_text_chunks
from groq import Groq

groq_client = Groq()
def process_url(urls):
    global index
    global chunk_texts_list
    loader = UnstructuredURLLoader(urls=urls)
    print("Loading data scrapped from urls")
    docs = loader.load()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    textsplitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "], chunk_size=350, chunk_overlap=30)
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
    D, I = index.search(query_embedding, k=8)
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
def process_doc_patient(query):
    global patient_index
    global patient_text_chunks
    documents = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, "patientexperience")
    if not os.path.exists(folder_path):
        return "Folder 'patientexperience' not found in the project directory."
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".docx"):  # Case-insensitive check
            full_path = os.path.join(folder_path, filename)
            print(f"Loading file: {full_path}")
            loader_patient = UnstructuredWordDocumentLoader(full_path)
            documents.extend(loader_patient.load())
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    textsplitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "], chunk_size=250, chunk_overlap=30)
    print("splitting documents into chunks")
    docs_chunks_patients = textsplitter.split_documents(documents)
    patient_text_chunks = [doc.page_content for doc in docs_chunks_patients]
    patientembeddings = model.encode(patient_text_chunks, convert_to_numpy=True)
    if patientembeddings.ndim == 1:
        patientembeddings = patientembeddings.reshape(1, -1)
    patientembeddings = patientembeddings.astype(np.float32)
    patientdimension = patientembeddings.shape[1]
    patient_index = fvd.IndexFlatIP(patientdimension)
    fvd.normalize_L2(patientembeddings)
    print("Adding documents to FAISS vector database")
    patient_index.add(patientembeddings)
    if patient_index is None or patient_text_chunks is None or len(patient_text_chunks) == 0:
        return "FAISS index is empty or chunk data is missing"
    query_embedding = model.encode([query], convert_to_numpy=True)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = query_embedding.astype(np.float32)
    fvd.normalize_L2(query_embedding)
    print("Print searching for similar chunks using semantic search")
    D, I = patient_index.search(query_embedding, k=8)
    if I is None or len(I[0]) == 0:
        return "No results found in the vector DB."
    matched_texts = [patient_text_chunks[idx] for idx in I[0] if 0 <= idx < len(patient_text_chunks)]
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


def process_doc_research(query):
    global research_index
    global research_text_chunks
    documents = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, "research")
    if not os.path.exists(folder_path):
        return "Folder 'research' not found in the project directory."
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".docx"):  # Case-insensitive check
            full_path = os.path.join(folder_path, filename)
            print(f"Loading file: {full_path}")
            loader_research = UnstructuredWordDocumentLoader(full_path)
            documents.extend(loader_research.load())
    if not documents:
        return "No research documents found or all were empty."
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    textsplitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "], chunk_size=250, chunk_overlap=30)
    print("splitting documents into chunks")
    docs_chunks_research = textsplitter.split_documents(documents)
    research_text_chunks = [doc.page_content for doc in docs_chunks_research]
    researchembeddings = model.encode(research_text_chunks, convert_to_numpy=True)
    if researchembeddings.ndim == 1:
        researchembeddings = researchembeddings.reshape(1, -1)
    researchembeddings = researchembeddings.astype(np.float32)
    researchdimension = researchembeddings.shape[1]
    research_index = fvd.IndexFlatIP(researchdimension)
    fvd.normalize_L2(researchembeddings)
    print("Adding documents to FAISS vector database")
    research_index.add(researchembeddings)
    if research_index is None or research_text_chunks is None or len(research_text_chunks) == 0:
        return "FAISS index is empty or chunk data is missing"
    query_embedding = model.encode([query], convert_to_numpy=True)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = query_embedding.astype(np.float32)
    fvd.normalize_L2(query_embedding)
    print("Print searching for similar chunks using semantic search")
    D, I = research_index.search(query_embedding, k=8)
    if I is None or len(I[0]) == 0:
        return "No results found in the vector DB."
    matched_texts = [research_text_chunks[idx] for idx in I[0] if 0 <= idx < len(research_text_chunks)]
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

def process_doc_doctor(query):
    global doctor_index
    global doctor_text_chunks
    documents = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, "doctorexperience")
    if not os.path.exists(folder_path):
        return "Folder 'doctorexperience' not found in the project directory."
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".docx"):  # Case-insensitive check
            full_path = os.path.join(folder_path, filename)
            print(f"Loading file: {full_path}")
            loader_doctor = UnstructuredWordDocumentLoader(full_path)
            documents.extend(loader_doctor.load())
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    textsplitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", " "], chunk_size=250, chunk_overlap=30)
    print("splitting documents into chunks")
    docs_chunks_doctor = textsplitter.split_documents(documents)
    doctor_text_chunks = [doc.page_content for doc in docs_chunks_doctor]
    doctorembeddings = model.encode(doctor_text_chunks, convert_to_numpy=True)
    if doctorembeddings.ndim == 1:
        doctorembeddings = doctorembeddings.reshape(1, -1)
    doctorembeddings = doctorembeddings.astype(np.float32)
    doctordimension = doctorembeddings.shape[1]
    doctor_index = fvd.IndexFlatIP(doctordimension)
    fvd.normalize_L2(doctorembeddings)
    print("Adding documents to FAISS vector database")
    doctor_index.add(doctorembeddings)
    if doctor_index is None or doctor_text_chunks is None or len(doctor_text_chunks) == 0:
        return "FAISS index is empty or chunk data is missing"
    query_embedding = model.encode([query], convert_to_numpy=True)
    if query_embedding.ndim == 1:
        query_embedding = query_embedding.reshape(1, -1)
    query_embedding = query_embedding.astype(np.float32)
    fvd.normalize_L2(query_embedding)
    print("Print searching for similar chunks using semantic search")
    D, I = doctor_index.search(query_embedding, k=8)
    if I is None or len(I[0]) == 0:
        return "No results found in the vector DB."
    matched_texts = [doctor_text_chunks[idx] for idx in I[0] if 0 <= idx < len(doctor_text_chunks)]
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

def process_llm_query(query):
    #function uses the llm directly
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": query}
        ],
        # max_tokens=2048
    )
    return response.choices[0].message.content
