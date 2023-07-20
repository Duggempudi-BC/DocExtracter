import streamlit as st
import openai 
import textract
from dotenv import load_dotenv
load_dotenv()
import openai
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from pinecone import Index
# importing required modules
import langchain
import pandas as pd 
import numpy as np
import PyPDF2
import os 
import requests
from dotenv import load_dotenv
load_dotenv()
pinecone.init(api_key=os.getenv("V_PINECONE_API_KEY") , environment='us-central1-gcp')
openai.api_key = os.getenv('OPENAI_API_KEY')
from langchain.document_loaders import Docx2txtLoader
import streamlit as st
from streamlit_chat import message
import streamlit.components.v1 as components
from pyngrok import ngrok






#extract text from the files
# textra = textract.process("data/demo.pdf").decode('utf-8').strip()



# #funtion to chunk the file 
# def chunk_token_splitter(text):
#     # the Token text Splitter
#     from langchain.text_splitter import TokenTextSplitter
#     text_splitter = TokenTextSplitter(chunk_size=50, 
#                                     chunk_overlap=0)
#     chunks = text_splitter.split_text(text)
#     return chunks, len(chunks)



# if __name__ == '__main__':
#     # create a pdf file object
#     pdfFileObj1 = open('data/demo.pdf', 'rb')
#     # pdfFileObj2 = open('pdf docs/class_12_physics/leph201.pdf', 'rb')
#     # pdfFileObj3 = open('pdf docs/class_12_physics/leph202.pdf', 'rb')
#     # pdfFileObj4 = open('pdf docs/class_12_physics/leph203.pdf', 'rb')
#     # pdfFileObj5 = open('pdf docs/class_12_physics/leph204.pdf', 'rb')
#     # pdfFileObj6 = open('pdf docs/class_12_physics/leph205.pdf', 'rb')
#     # pdfFileObj7 = open('pdf docs/class_12_physics/leph206.pdf', 'rb')

#     #pass in text from pdf to get text
#     all_pdf_content = textra

#     #Split into tokens
#     chunked_pdf_content = chunk_token_splitter(all_pdf_content)[0]


#     pdfFileObj1.close()
#     # pdfFileObj2.close()
#     # pdfFileObj3.close()
#     # pdfFileObj4.close()
#     # pdfFileObj5.close()
#     # pdfFileObj6.close()
#     # pdfFileObj7.close() 
    




#funtion for embedding text
def text_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']





# index_name = "text-index"
# pinecone.create_index(index_name, dimension=384)

# def chunks_to_index(chunked_data):

#     #creating embedding to store embedded text and metadata to store text for reference
#     chunk_list = chunked_data
#     # embeddings =[]
#     # metadata = []
#     cnt= 1

#     for chunk in chunk_list[0:20]:

#         embedded_text = text_embedding(chunk) # embedding the text inside each chunk
#         metadata = {cnt: chunk}
    
        
#         index.upsert( [ (f"{cnt}", embedded_text , metadata)])
#         cnt =cnt+ 1

        
# chunks_to_index(chunked_pdf_content)




#funtion to retrive query
index = pinecone.Index("docbot")
gpt_meta = ''
def query_user_data(query):

    query_vec = text_embedding(query)

    #pinecone retrival query
    out = index.query(              
    vector=query_vec,
    top_k=3,
    include_values=True,
    include_metadata=True,
    score= 0.8,
    )

    #filtering the similirity search by increasing the cosine threshold
    filtered_results = [result for result in out['matches'] if result['score'] > 0.8]

    #to extract the metadata stored in pinecone
    gpt_meta = ''
    for match in filtered_results:
        for i ,metadata in match["metadata"].items():
            gpt_meta = f"{gpt_meta} {metadata}"

    

    return gpt_meta



#funtion to get the response for user query 
def send_message(message , conversations):
    

    conversations.append({'role':'user', 'content':f"{message}"})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= conversations,
        temperature = 0.0
        
    )

    if len(conversations) == 16:
        conversations.pop(1)
        conversations.pop(2)
        
    reply = response.choices[0].message.content
    conversations.append({'role':'user', 'content':f"{reply}"})
    
    return reply




st.title("Document BOT")


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

response_container = st.container()

textcontainer = st.container()

with textcontainer:

    def get_text():
        input_text = st.text_input("User: ")
        return input_text 


    user_input = get_text()


if user_input:
    test = query_user_data(user_input)
else:
    test=''
conversations = [{'role':'system', 'content':f"""
                  
First you Greet me
You first task is to analyse whether the given below CONTEXT is empty or not 
if it is empty then strictly reply "Sorry , I don't have the information realted your query ! "

CONTEXT = {test}

Dont use your own knowledge to answer the question stricly follow the CONTEXT only
Summarize and Give answer based on the question strictly from the CONTEXT given below

Question= {user_input}
CONTEXT = {test}




your response should be polite, neat and friendly

"""} ]



with response_container:

    if user_input:
        output = send_message(user_input , conversations)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)

    if st.session_state['generated']:

        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            
    
# from pyngrok import ngrok 
# !ngrok authtoken [Enter your authtoken here]
# !nohup streamlit run app.py & 

# url = ngrok.connect(port = 8501)
# url #generates our URL

# !streamlit run --server.port 80 app.py >/dev/null #used for starting our server

    
            
