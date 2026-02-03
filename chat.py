import streamlit as st
from dotenv import load_dotenv

from rag.llm import get_ai_response

st.set_page_config(page_title="사업수행팀 교육문서", page_icon="🍊")

st.title("교육문서")
st.caption("사업수행팀 교육문서")

# 환경변수를 불러줌
load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

print(st.session_state.message_list)

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
         st.write(message["content"])


if user_question := st.chat_input(placeholder="placeholder"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role":"user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다."):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):

            ai_message = st.write(ai_response)
            st.session_state.message_list.append({"role":"ai", "content": ai_message})
