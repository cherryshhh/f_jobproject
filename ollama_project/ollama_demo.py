import ollama
import streamlit as st

client = ollama.Client(host = "http://localhost:11434")

#init
if 'message' not in st.session_state:
    st.session_state['message'] = []

#title
st.title("deepseek-r1:7b")

st.divider()

#user input
prompt = st.chat_input("请输入你的问题")

#judge
if prompt :
    #add history
    st.session_state['message'].append({"role": "user", "content": prompt})

    for message in st.session_state['message']:
        st.chat_message(message['role']).markdown(message['content'])

    with st.spinner("思考中..."):
        response = client.chat(
            model='deepseek-r1:7b',
            messages=[{"role":"user", "content": prompt}]
        )
    st.session_state['message'].append({"role": "assistant", "content": response['message']['content']})
    st.chat_message("assistant").markdown(response['message']['content'])