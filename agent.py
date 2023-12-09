__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from sklearn.cluster import KMeans
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import numpy as np
from langchain.chains import ConversationalRetrievalChain
import logging
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


logging.basicConfig(level=logging.INFO) 
# Initialize global variables
global_chromadb = None
global_documents = None
global_short_documents = None 

# Initialize the memory outside the function so it persists across different calls
conversation_memory = ConversationBufferMemory(
    memory_key="chat_history",
    max_len=50,
    input_key="question",
    output_key="answer",
    return_messages=True,
)

# Function to reset global variables
def reset_globals():
    global global_chromadb, global_documents, global_short_documents
    global_chromadb = None
    global_documents = None
    global_short_documents = None
    # Reset the conversation memory
    if conversation_memory:
        conversation_memory.clear()

def init_chromadb(openai_api_key):
    global global_chromadb, global_short_documents
    if global_chromadb is None and global_short_documents is not None:
        global_chromadb = Chroma.from_documents(documents=global_short_documents, embedding=OpenAIEmbeddings(openai_api_key=openai_api_key))

def process_and_cluster_captions(captions, openai_api_key, num_clusters=12):
    global global_documents, global_short_documents 
    logging.info("Processing and clustering captions")
    
    # Log the first 500 characters of the captions to check their format
    logging.info(f"Captions received (first 500 characters): {captions[0].page_content[:500]}")
    caption_content = captions[0].page_content

    # Ensure captions is a string before processing
    if not isinstance(caption_content, str):
        logging.error("Captions are not in the expected string format")
        return []
    
    # Create longer chunks for summary
    summary_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])
    summary_docs = summary_splitter.create_documents([caption_content])

    # Create shorter chunks for QA
    qa_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])
    qa_docs = qa_splitter.create_documents([caption_content])
    
    # Process for summary
    summary_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key).embed_documents([x.page_content for x in summary_docs])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(summary_embeddings)
    closest_indices = [np.argmin(np.linalg.norm(summary_embeddings - center, axis=1)) for center in kmeans.cluster_centers_]
    representative_docs = [summary_docs[i] for i in closest_indices]

    # Store documents globally
    global_documents = summary_docs  # For summary
    global_short_documents = qa_docs  # For QA

    init_chromadb(openai_api_key)  # Initialize database with longer chunks
    return representative_docs


def generate_summary(representative_docs, openai_api_key, model_name):
    logging.info("Generating summary")
    llm4 = ChatOpenAI(model_name=model_name, temperature=0.2, openai_api_key=openai_api_key)

    # Concatenate texts for summary
    summary_text = "\n".join([doc.page_content for doc in representative_docs])

    summary_prompt_template = PromptTemplate(
        template=(
            "Create a concise summary of a podcast conversation based on the text provided below. The text consists of selected, representative sections from different parts of the conversation. "
            "Your task is to synthesize these sections into a single cohesive and concise summary. Focus on the overarching themes and main points discussed throughout the podcast. "
            "The summary should give a clear and complete understanding of the conversation's key topics and insights, while omitting any extraneous details. It should be engaging and easy to read, ideally in one or two paragraphs. Keep it short where possible"
            "\n\nSelected Podcast Sections:\n{text}\n\nSummary:"
        ),
        input_variables=["text"]
    )
    # Load summarizer chain
    summarize_chain = load_summarize_chain(llm=llm4, chain_type="stuff", prompt=summary_prompt_template)

    # Run the summarizer chain
    summary = summarize_chain.run([Document(page_content=summary_text)])

    logging.info("Summary generation completed")
    return summary


def answer_question(question, openai_api_key, model_name):
    llm4 = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=openai_api_key)
    global global_chromadb, global_short_documents

    if global_chromadb is None and global_short_documents is not None:
        init_chromadb(openai_api_key, documents=global_short_documents)
    
    logging.info(f"Answering question: {question}")
    chatTemplate = """
    You are an AI assistant tasked with answering questions based on context from a podcast conversation. Use the provided context and relevant chat messages to answer. If unsure, say so. Keep your answer to three sentences or less, focusing on the most relevant information.
    Chat Messages (if relevant): {chat_history}
    Question: {question} 
    Context from Podcast: {context} 
    Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question", "chat_history"],template=chatTemplate)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm4, 
        chain_type="stuff", 
        retriever=global_chromadb.as_retriever(search_type="mmr", search_kwargs={"k":12}),
        memory=conversation_memory,
        #return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
    )
    # Log the current chat history
    current_chat_history = conversation_memory.load_memory_variables({})
    logging.info(f"Current Chat History: {current_chat_history}")
    response = qa_chain({"question": question}) 
    logging.info(f"this is the result: {response}")
    output = response['answer']    

    return output