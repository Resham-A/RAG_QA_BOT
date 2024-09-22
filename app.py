import streamlit as st
import pickle
import os
import torch
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.langchain import LangchainEmbedding


  
def main():
    st.header('RAG QA BOT')

    load_dotenv()  
    ##upload a file
    pdf = st.file_uploader('upload pdf', type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        st.write(pdf.name)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        #st.write(chunks)
        
        #embeddings
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                db = pickle.load(f)
        else:
            embeddings = LangchainEmbedding(
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
           
            db = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(db, f)

    
       
       #user Query
        query = st. text_input("Ask question:")
    
        if query:
            docs = db.similarity_search(query=query, k=3)

            llm = HuggingFaceLLM(tokenizer_name="Undi95/Meta-Llama-3-70B-hf", model_kwargs={"temperature":0,"max_length":100})
        
            chain=load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)





if __name__ == '__main__':
    main()
