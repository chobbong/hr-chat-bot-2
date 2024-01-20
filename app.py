import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
import openai
import streamlit as st
from streamlit_chat import message
import plotly.express as px
from openai import OpenAI

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=API_KEY)

# 임베딩 모델을 사용하여 텍스트의 임베딩을 얻는 함수
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

# folder_path와 file_name을 결합하여 file_path 생성
folder_path = './data'
file_name = 'embedding.csv'
file_path = os.path.join(folder_path, file_name)

# embedding.csv 파일이 존재하는지 확인하고 로드 또는 생성
if os.path.isfile(file_path):
    print(f"{file_name} 파일이 존재합니다.")
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(ast.literal_eval)
else:
    txt_files = [file for file in os.listdir(folder_path) if file.endswith('.txt')]
    data = []
    for file in txt_files:
        txt_file_path = os.path.join(folder_path, file)
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append(text)

    df = pd.DataFrame(data, columns=['text'])
    df['embedding'] = df['text'].apply(lambda x: get_embedding(x, model="text-embedding-ada-002"))
    df.to_csv(file_path, index=False, encoding='utf-8-sig')

# 코사인 유사도 계산 함수
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

# 질문에 대한 후보 답변 반환 함수
def return_answer_candidate(df, query):
    query_embedding = get_embedding(query, model="text-embedding-ada-002")
    df["similarity"] = df.embedding.apply(lambda x: cos_sim(np.array(x), np.array(query_embedding)))
    top_three_doc = df.sort_values("similarity", ascending=False).head(3)
    return top_three_doc

## 챗봇 프롬프트 생성 함수
def create_prompt(df, query):
    result = return_answer_candidate(df, query)
    docs = [f"doc {i+1}: {doc['text']}" for i, doc in result.iterrows()]
    system_role = (f"""You are an artificial intelligence language model named "정채기" that specializes in summarizing and answering documents about Seoul's youth policy, developed by developers 조윤서. 
    You need to take a given document and return a very detailed summary of the document in the query language.
    Here are the documents:
    {' '.join(docs)}
    You must return in Korean. Return an accurate answer based on the document.
    """)
    user_content = f"""User question: "{query}"."""
    messages = [{"role": "system", "content": system_role}, {"role": "user", "content": user_content}]
    return messages

# 챗봇 답변 생성 함수
def generate_response(messages):
     result = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=messages,
          temperature=0.4,
          max_tokens=500
     )
     return result.choices[0].message.content

# Streamlit UI 설정
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('정책을 보세요!', '', key='input')
    submitted = st.form_submit_button('Send')

if submitted and user_input:
    prompt = create_prompt(df, user_input)
    chatbot_response = generate_response(prompt)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(chatbot_response)

# 사용자의 질문과 첫봇의 답변을 순차적으로 화면에 출력
if st.session_state['generated']:
     for i in reversed(range(len(st.session_state['generated']))):
          message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
          message(st.session_state['generated'][i], key=str(i))


