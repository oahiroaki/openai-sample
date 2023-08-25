"""
main.py
"""
import os
from dotenv import load_dotenv
import streamlit as st
import openai

# .envファイルの読み込み
load_dotenv()
# OpenAI APIキーの設定
openai.api_key = os.environ["OPENAI_API_KEY"]

# Title
st.title("streamlit sample")

# side
st.sidebar.title("Options")

# サイドバーにオプションボタンを設置
user_select_model = st.sidebar.radio("Choose a model:", ["gpt-3.5-turbo"])

# サイドバーにボタンを設置
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# サイドバーにスライダーを追加し、temperatureを0から2までの範囲で選択可能にする
# 初期値は0.0、刻み幅は0.1とする
user_select_temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"

def response_chatgpt(user_message: str):
    """
    ChatGPTのレスポンスを取得
    Args:
        user_msg (str): ユーザーメッセージ。
    """
    chat_response = openai.ChatCompletion.create(
        model=user_select_model,
        temperature=user_select_temperature,
        messages=[
            {"role": "user", "content": user_message},
        ],
        stream=True,
    )
    return chat_response


# チャットログを保存したセッション情報を初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []


user_msg = st.chat_input("ここにメッセージを入力")

if user_msg:
    # 以前のチャットログを表示
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.write(chat["msg"])

    # 最新のメッセージを表示
    with st.chat_message(USER_NAME):
        st.write(user_msg)

    # アシスタントのメッセージを表示
    response = response_chatgpt(user_msg)
    with st.chat_message(ASSISTANT_NAME):
        assistant_response_area = st.empty()
        assistant_message = ""
        for chunk in response:
            # 回答を逐次表示
            assistant_message += chunk["choices"][0]["delta"].get("content", "")
            assistant_response_area.write(assistant_message)

    # セッションにチャットログを追加
    st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
    st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_message})
