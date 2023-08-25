
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader

from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# load_dotenv()　Home(main)で実行されるため不要

def init_page():
    st.set_page_config(
        page_title="Yoututbe Summarizer"
    )
    st.header("Youtube Summarizer")
    st.sidebar.title("Youtube Summarizer Options")
    st.session_state.costs = []
    
def get_url_input():
    url = st.text_input("Youtube URL: ", key="inputYoutubeSummarizer1")
    return url
    
def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    return ChatOpenAI(temperature=0, model_name=model_name)

def get_document(url):
    with st.spinner("Fetching Content ..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language=["en", "ja"]
        )
        return loader.load()
    
def summarize(llm, docs):
    prompt_template = """Write a concise Japanese summary of the following transcript of Youtube Video.

============
    
{text}

============

ここから日本語で書いてね
必ず3段落以内の200文字以内で簡潔にまとめること:
"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    
    with get_openai_callback() as cb:
        chain = load_summarize_chain( 
            llm,
            chain_type="stuff",
            verbose=True,
            prompt=PROMPT
        )
        response = chain({"input_documents": docs}, return_only_outputs=True)
    return response["output_text"], cb.total_cost
    
def main():
    init_page()
    llm = select_model()
    
    container = st.container()
    response_container = st.container()
    
    with container:
        url = get_url_input()
        if url:
            document = get_document(url)
            with st.spinner("ChatGPT is typing ..."):
                output_text, cost = summarize(llm, document)
            st.session_state.costs.append(cost)
        else:
            output_text = None
    
    if output_text:
        with response_container:
            st.markdown("## 要約")
            st.write(output_text)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(document)
            
    costs = st.session_state.get("costs", [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")
    
if __name__ == '__main__':
    main()