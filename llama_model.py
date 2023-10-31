import streamlit as st
from streamlit_chat import message
import sentencepiece 
import whisper
from pytube import YouTube
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.llms import CTransformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import time
import os

os.environ["REPLICATE_API_TOKEN"] = "r8_CbaXZv51Qkp4v91Cuc5pA5XHwTWKO8h00q1BE"
model = whisper.load_model("base")

#model and tokenizer loading
checkpoint = "C:/Users/shrit/OneDrive/Documents/Long-descriptive-videos/LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)
pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer = tokenizer,
        max_length = 500, 
        min_length = 50)

st.set_page_config(
    layout="wide"
)

extracted_text = None

def transcribe_audio(url):
    global extracted_text
    yt = YouTube(url)
    streams = yt.streams.filter(only_audio=True)
    stream = streams.first()
    stream.download(filename = "demo1.mp4")
    tran = model.transcribe("demo1.mp4")
    extracted_text = tran["text"]
    with open("extracted_text.txt", "w") as f:
        f.write(extracted_text)

    return extracted_text


#LLM pipeline
def llm_pipeline(input_text):
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result
    #instruction = transcribe_audio(url)
    #input_prompt = f"Given a long text as instruction, summarize it with utmost precision and context.\n\n### Instruction:\n{instruction}\n\n### Response:"
    #enerated_text = pipe_sum(input_prompt, do_sample=True)[0]['generated_text']
    #print("Response", generated_text)


def chat_llm():
    
    loader = TextLoader("extracted_text.txt")
    documents = loader.load()

    #split text into chunks
    text_splitter  = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    #create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':"cpu"})

    #vectorstore
    vector_store = FAISS.from_documents(text_chunks,embeddings)

    #create llm
    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",
                    config={'max_new_tokens':128,'temperature':0.01})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)
    return chain


def main():


    # Set the title and background color
    st.title(" SummarizeChatVision🎥")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('YouTube Video Summarizer and Chatbot❤️')
    st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

    # Expander for app details
    with st.expander("About the App"):
        st.write("This app allows you to interact with the content of the video!.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start.")

    # Input box for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL")
    time.sleep(10) 
    transcript = transcribe_audio(youtube_url)
    summary = llm_pipeline(transcript)

    # Submit button
    if st.button("Submit"):
        #start_time = time.time()  # Start the timer
        
        #end_time = time.time()  # End the timer
        #elapsed_time = end_time - start_time

        # Display layout with 2 columns
        col1, col2 = st.columns([1,1])

        # Column 1: Video view
        with col1:
            st.video(youtube_url)
            #st.header("Video Transcript")
            #st.write(output)
            #st.success(transcript.split("\n\n[INST]")[0])
            #st.write(f"Time taken: {elapsed_time:.2f} seconds")

        with col2 :
            st.header("Video Summary")
            start_time = time.time()
            st.info("Summarization Complete")
            st.success(summary)
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.write(f"Time taken: {elapsed_time:.2f} seconds")


    def conversation_chat(query):
        chain = chat_llm()
        
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    #initialize_session_state
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about about this video 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    #conversation_chat = conversation_chat #changes

    reply_container = st.container()
    container = st.container()

    with container:
        user_input = st.text_input("Question:", placeholder="Ask about the video")
        submit_button = st.button('Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with reply_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

            
if __name__ == "__main__":
    main()