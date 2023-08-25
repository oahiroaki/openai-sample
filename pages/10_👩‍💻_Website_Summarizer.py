"""
main.py
"""
import os
from typing import List
from dotenv import load_dotenv

from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import (HumanMessage, AIMessage)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
import openai
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

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
# clear_button = st.sidebar.button("Clear Conversation", key="clear")

# サイドバーにスライダーを追加し、temperatureを0から2までの範囲で選択可能にする
# 初期値は0.0、刻み幅は0.1とする
user_select_temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"
COLLECTION_NAME = "my_collection"
SUMMARY_CONTENT_LIMIT = 400

def initialize():
    """
    初期表示処理
    """

    # st.set_page_config(
    #     page_title="streamlit sample",
    #     page_icon="🤗"
    # )
    # メッセージがURL形式かをチェック
    # if validate_url(user_message):
    #     content = get_content(user_message)
    #     user_message = build_web_summary_prompt(content)

    # chat_response = openai.ChatCompletion.create(
    #     model=user_select_model,
    #     temperature=user_select_temperature,
    #     messages=[
    #         {"role": "user", "content": user_message},
    #     ],
    #     stream=True,
    # )

def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            # SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    return ChatOpenAI(temperature=0, model_name=model_name)


def get_url_input():
    url = st.text_input("URL: ", key="input")
    return url

def get_content(url: str):
    """
    Webページのコンテンツ取得
    Args:
        ulr (str): URL
    """
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        if soup.main:
            return soup.main.get_text()
        elif soup.article:
            return soup.article.get_text()
        else:
            return soup.body.get_text()
    except:
        st.write('something wrong')
        return None

def build_web_summary_prompt(content: str):
    """
    Webサイト要約用のプロンプト作成
    Args:
        content (str): Webページのコンテンツ
    """
    return f"""以下はとあるWebページのコンテンツである。内容を{SUMMARY_CONTENT_LIMIT}程度でわかりやすく要約してください。

========

{content[:1000]}

========

日本語で書いてね！
"""

def validate_url(url: str):
    """
    URLチェック
    Args:
        url (str): URL
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def page_chat(openai_api_key):
    """_summary_
    チャットページ
    """

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

def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost

def main():
    initialize()

    llm = select_model()
    init_messages()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write('Please input valid url')
            answer = None
        else:
            content = get_content(url)
            if content:
                prompt = build_web_summary_prompt(content)
                st.session_state.messages.append(HumanMessage(content=prompt))
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = get_answer(llm, st.session_state.messages)
                st.session_state.costs.append(cost)
            else:
                answer = None

    if answer:
        with response_container:
            st.markdown("## Summary")
            st.write(answer)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(content)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()
