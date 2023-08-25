"""
main.py
"""
import os
from typing import List
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import (HumanMessage, AIMessage)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
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

    # costs = st.session_state.get('costs', [])
    # st.sidebar.markdown("## Costs")
    # st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    # for cost in costs:
    #     st.sidebar.markdown(f"- ${cost:.5f}")


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
    #qdrant = load_qdrant()
    #qdrant.add_texts(pdf_text)
    Qdrant.from_texts(
        pdf_text,
        OpenAIEmbeddings(),
        path = QDRANT_PATH,
        collection_name = COLLECTION_NAME,
    )


def build_qa_model(llm):
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        # "mmr",  "similarity_score_threshold" などもある
        search_type="similarity",
        # 文書を何個取得するか (default: 4)
        search_kwargs={"k":10}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

def ask(qa, query):
    with get_openai_callback() as cb:
        # query / result / source_documents
        answer = qa(query)

    return answer, cb.total_cost

def select_model():
    model = st.sidebar.radio("Choose a model:", ("gpt-3.5-turbo", "gpt-3.5-turbo-16k"))
    st.session_state.model_name = model

    # 300: 本文以外の指示のトークン数 (以下同じ)
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)

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
    """
    PDF問い合わせページ
    """

    st.title("Ask My PDF(s)")

    llm = select_model()
    container = st.container()
    response_container = st.container()

    with container:
        query = st.text_input("Query: ", key = "input")
        if not query:
            answer = None
        else:
            qa = build_qa_model(llm)
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = ask(qa, query)
                # st.session_state.costs.append(cost)
            else:
                answer = None

        if answer:
            with response_container:
                st.markdown("## Answer")
                st.write(answer)


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
            # モデル
            model_name = "text-embedding-ada-002",
            # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
            chunk_size = 300,
            chunk_overlap = 0,
        )
        return text_splitter.split_text(text)
    else:
        return None

if __name__ == '__main__':
    # .envファイルの読み込み
    load_dotenv()
    # 初期化処理
    initialize(os.environ["OPENAI_API_KEY"])
