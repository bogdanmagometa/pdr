import os
import streamlit as st
import time
from langserve import RemoteRunnable
import dotenv

dotenv.load_dotenv(override=False);

def get_response(user_input):
    hostname = os.getenv('PDR_MS_HOSTNAME')
    port = os.getenv('PDR_MS_PORT')
    
    qa_chain = RemoteRunnable(f"http://{hostname}:{port}/pdr_qa")
    response = qa_chain.stream({'input': user_input})

    for chunk in response:
        if 'answer' in chunk:
            answer = chunk['answer']
        else:
            answer = ''
        if 'context' in chunk:
            for i in range(len(chunk['context'])):
                print(chunk['context'][i].page_content)
        yield answer

st.set_page_config(page_title="ПДР чатбот")
st.title("ПДР чатбот")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for sender, message in st.session_state.chat_history:
    st.chat_message(sender).write(message)

user_input = st.chat_input("Задай запитання про дорожній рух", key="input")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(("user", user_input))

    response_stream = get_response(user_input)

    response = st.chat_message("assistant").write_stream(response_stream)
    st.session_state.chat_history.append(("assistant", response))
