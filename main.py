"""
main.py
"""
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import (HumanMessage, AIMessage)
from typing import Any, Dict, List

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"

def initialize(openai_api_key: str):
    """
    初期表示処理
    """

    st.set_page_config(
        page_title="streamlit sample",
        page_icon="🤗"
    )

    # サイドバー
    st.sidebar.title("Navigation")
    # サイドバー：ページ切り替えナビゲーション
    user_selection = st.sidebar.radio("Go to", ["Chat", "PDF Upload", "Ask My PDF(s)"])

    if user_selection == "PDF Upload":
        page_pdf_upload_and_build_vector_db()
    elif user_selection == "Ask My PDF(s)":
        page_ask_my_pdf()
    else:
        page_chat(openai_api_key)


def page_pdf_upload_and_build_vector_db():
    """_summary_
    PDF読み込みページ
    """
    return

def page_ask_my_pdf():
    """_summary_
    PDF問い合わせページ
    """
    return

def page_chat(openai_api_key):
    """_summary_
    チャットページ
    """

    st.title("Chat")

    # サイドバー：モデル選択
    user_select_model = st.sidebar.radio("Choose a model:", ["gpt-3.5-turbo"])
    # サイドバー：会話履歴削除
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    # サイドバー：temperatureを0から2までの範囲で選択
    user_select_temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    
    # セッション情報がない場合、チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # チャット履歴の表示
    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message(ASSISTANT_NAME):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message(USER_NAME):
                st.markdown(message.content)
        else:
            st.write(f"System message: {message.content}")

    if user_message := st.chat_input("聞きたいことを入力してください"):
        # セッション保存
        st.session_state.messages.append(HumanMessage(content=user_message))
        with st.chat_message(USER_NAME):
            st.markdown(user_message)
        # OpenAI API通信クラス初期化
        llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=user_select_model,
            temperature=user_select_temperature)
        with st.spinner("ChatGPT is typing ..."):
            response = llm(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))
        with st.chat_message(ASSISTANT_NAME):
            st.markdown(response.content)


if __name__ == '__main__':
    # .envファイルの読み込み
    load_dotenv()
    # 初期化処理
    initialize(os.environ["OPENAI_API_KEY"])
