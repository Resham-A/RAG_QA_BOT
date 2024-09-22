
# RAG_QA_BOT

Retrieval-Augmented Generation(RAG) Model for a Question Answering(QA) bot for a business.
RAG is a way for augmenting LLM data with additional information.
RAG-Based model that can handle questions related to a provided document or dataset.
Here is the QA bot which is allowing users to upload PDF documents and ask questions. It then stores embeddings and provide real-time answers to user queries.






## Prerequisites:
langchain_community
langchain_core
langserve
PyPDF2
python-dotenv
streamlit
streamlit-extras
fastapi
pandas
faiss-CPU

## Installation

Set up environment by running the following commands :


conda create -p myenv python==3.12

conda activate myenv/
 
Install following Packages:

pip install -r requirements.txt
pip install sentence_transformer
pip install llama_index
huggingface-cli login

    


## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`HUGGINGFACEHUB_API_TOKEN`="my api token"


## Deployment

To deploy this project run

streamlit run app.py

