"""
main.py
"""
import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.schema import (HumanMessage, AIMessage)
import streamlit as st

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"
QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection"

def initialize(openai_api_key: str):
    """
    初期表示処理
    """

    st.set_page_config(
        page_title="Chat",
        page_icon="🤗"
    )

    st.title("Chat")

    # サイドバー：モデル選択
    user_select_model = st.sidebar.radio("Choose a model:", ["gpt-3.5-turbo-16k", "gpt-3.5-turbo"])
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
