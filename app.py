from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from dotenv import load_dotenv
from design import bot_template , user_template ,css
import qdrant_client
from langchain.chains import RetrievalQA
import os



def get_vector_store():
    embedding = OpenAIEmbeddings()
    client = qdrant_client.QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv('QDRANT_API_KEY')
    )
    vector_store = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embedding
    )
    return vector_store


def main():
    load_dotenv()
    st.set_page_config(page_title= "5th Sem Sol" , page_icon=":books:")
    st.header("Ask your problem from any subject ")
    st.write(css , unsafe_allow_html=True)
    vector_store= get_vector_store()
    qa= RetrievalQA.from_chain_type(
        llm = OpenAI(),
        chain_type="stuff" ,
        retriever=vector_store.as_retriever()
    )

    user_input = st.text_input("Please enter your question")
    if user_input:
        st.write(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)
        with st.spinner("Getting response"):
         reponse = qa.run(user_input)
         st.write(bot_template.replace("{{MSG}}", reponse), unsafe_allow_html=True)

if __name__ == "__main__":
    main()