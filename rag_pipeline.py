import os
import openai
from openai import OpenAI
import streamlit as st

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
#api_key = os.environ['ZZZ_API_KEY']
api_key = st.secrets["ZZZ_API_KEY"]

client = OpenAI(
    base_url = 'https://api.zhizengzeng.com/v1',
    api_key = api_key,
)

#Load the data
from langchain_community.document_loaders import PyPDFLoader

file_path = "employee_handbook.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

#clean the data
raw_text = "\n".join([d.page_content for d in docs])
clean_text = raw_text.replace("\x01", " ")

#Chunk the data
def chunk_text(text, chunk_size = 400, over_lap = 100):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start+chunk_size, n)
        chunk = text[start:end].strip()

        if chunk:
            
            chunks.append(chunk)
        if end == n:
            break
        start = end-over_lap

    return chunks

#embeddings
from sentence_transformers import SentenceTransformer

embedding = SentenceTransformer("BAAI/bge-base-zh-v1.5")
def embed(texts):
    vecs = embedding.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs

chunks = chunk_text(clean_text, chunk_size = 400, over_lap = 100)
#print(len(chunks))
chunks_embeddings = embed(chunks)
#vector database
import faiss
import numpy as np

index = faiss.IndexFlatL2(chunks_embeddings.shape[1])
index.add(chunks_embeddings)
#retrieve the relevant information
def retrieve(query, top_k = 3):
    query_vec = embed([query]).astype("float32")
    dist, idx = index.search(query_vec, top_k)
    results = []
    for rank, (d, i) in enumerate(zip(dist[0], idx[0])):
        results.append({
            "rank": rank+1,
            "chunk_id": int(i),
            "distance": float(d),
            "text": chunks[i]
        })
    return results

query = "年假怎么休？"
results = retrieve(query)
#print(results)

#re-prompt agent
def reprompt_agent(query):
    """
    Transfer the original query to a more complete question for further retrieve
    """
    system_msg = """
    你是中文re-prompt agent，帮助用户把模糊或短的问题写完整，用于后续RAG检索
    """
    user_msg = f"""
    你会收到用户关于企业规章&人力资源管理相关的query。请先理解用户query，并站在员工立场把query补充成完整，
    明确的问题，用于后续RAG检索。 仅输出改善后的中文query。
    用户的query:{query}
    """
    response = client.chat.completions.create(
        messages = [
    {"role":"system", "content":system_msg},
    {"role":"user", "content": user_msg},
        ],
        model = "gpt-5-nano",
    )

    new_query = response.choices[0].message.content
    return new_query

def LLM_using_RAG(query,model):
    #first, we reprompt the query
    new_query = reprompt_agent(query)

    #print("STEP 1: Reprompt =======")
    #print(new_query)
    #next, retrieve
    retrieved_results = retrieve(new_query)
    #print("STEP 2: Retrive Relevant Info ======")
    
    #print(retrieved_results)
    system_prompt = f"""
    你是企业的HR助手，善于回答员工关于企业相关问题
    """

    user_prompt = f"""
    你会收到用户的query和一份参考资料。
    要求：
    1. 你有你自己的常识，你可以选择用或者不用参考资料
    2. 请以HR的口吻来做回答
    3. 如果用了参考资料，请在回答中包含参考资料的原文和reference

    用户的query:{query}
    参考资料:{retrieved_results}
    """
    response = client.chat.completions.create(
        messages = [
            {"role":"system", "content":system_prompt},
            {"role":"user", "content": user_prompt},
        ],
        model = model,
    )

    #print("STEP 3: LLM use results to generate Answer ====")
    return response.choices[0].message.content
    
query = "年假怎么休？"
response = LLM_using_RAG(query, model = "gpt-5-nano")
