"""
main.py
"""
import os
from typing import List
from dotenv import load_dotenv

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.chat_models import ChatOpenAI
from langchain.schema import (HumanMessage, AIMessage)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from PyPDF2 import PdfReader

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

def load_qdrant():
    client = QdrantClient(path = QDRANT_PATH)
    # すべてのコレクション名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name = COLLECTION_NAME,
            vectors_config = VectorParams(size = 1536, distance = Distance.COSINE),
        )
        print('collection created')

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=OpenAIEmbeddings()
    )

def build_vector_store(pdf_text):
    qdrant = load_qdrant()
    qdrant.add_texts(pdf_text)


def page_pdf_upload_and_build_vector_db():
    """
    PDF読み込みページ
    """

    st.title("PDF Upload")
    container = st.container()
    with container:
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(pdf_text)


def page_ask_my_pdf():
    """_summary_
    PDF問い合わせページ
    """
    return


def get_pdf_text() -> List[str]:
    """
    PDFのテキストを読み取ってチャンクに分割する

    Returns:
        チャンクのリスト
    """

    uploaded_file = st.file_uploader(label="Upload your PDF here😇", type="pdf")
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=st.session_state.emb_model_name,
            # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
            # 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
            # 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
            chunk_size=250,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)
    else:
        return None

if __name__ == '__main__':
    # .envファイルの読み込み
    load_dotenv()
    # 初期化処理
    initialize(os.environ["OPENAI_API_KEY"])
