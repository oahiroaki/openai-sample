"""
main.py
"""
import os
from dotenv import load_dotenv
import streamlit as st
import openai

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"

def initialize():
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
        page_chat()


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

def page_chat():
    """_summary_
    チャットページ
    """

    st.title("Chat")

    # サイドバー：モデル選択
    user_select_model = st.sidebar.radio("Choose a model:", ["gpt-3.5-turbo"])

    # サイドバー：会話履歴削除
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # サイドバーにスライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.1とする
    user_select_temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

    if user_message := st.chat_input("ここにメッセージを入力"):
        # 最新のメッセージを表示
        with st.chat_message(USER_NAME):
            st.write(user_message)

        # AI問い合わせ
        response = openai.ChatCompletion.create(
            model=user_select_model,
            temperature=user_select_temperature,
            messages=[
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )

        with st.chat_message(ASSISTANT_NAME):
            assistant_response_area = st.empty()
            assistant_message = ""
            for chunk in response:
                # 回答を逐次表示
                assistant_message += chunk["choices"][0]["delta"].get("content", "")
                assistant_response_area.write(assistant_message)

if __name__ == '__main__':
    # .envファイルの読み込み
    load_dotenv()
    # OpenAI APIキーの設定
    openai.api_key = os.environ["OPENAI_API_KEY"]

    initialize()
