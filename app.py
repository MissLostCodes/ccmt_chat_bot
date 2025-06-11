#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[10]:


import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
import streamlit_analytics
import pinecone
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore


# In[13]:


# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "index1"  


# ## load pdfs 

# In[50]:


# Load all PDFs
brochure_docs = PyPDFLoader("ccmt_info.pdf").load()
flowchart_docs = PyPDFLoader("ccmt_flowcharts_2.pdf").load()
fee_docs = PyPDFLoader("fee_table2.pdf").load()

# Add metadata (tags)
for doc in fee_docs:
    doc.metadata["source"] = "fee_clean"   

for doc in brochure_docs:
    doc.metadata["source"] = "brochure"

for doc in flowchart_docs:
    doc.metadata["source"] = "flowchart"


# ### chunking

# In[51]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(flowchart_docs + fee_docs + brochure_docs )



# In[ ]:





# In[52]:


import re

def clean_text(text):
    # Remove invalid surrogate characters
    text = text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    # Optional: Remove emojis or other symbols
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

# Apply to chunks
for doc in chunks:
    doc.page_content = clean_text(doc.page_content)


# In[53]:


print("Type of variable:", type(chunks))
print()
print("Type of each object inside the list:", type(chunks[0]))
print()
print("Total number of documents inside list:", len(chunks))
print()
print("* Content of first chunk:", chunks[60])
print()
print("* Content of second chunk:", chunks[125])
print()
print("* Content of second chunk:", chunks[1])


# In[54]:


embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001" , google_api_key=GOOGLE_API_KEY)


index_name = PINECONE_INDEX_NAME



# # pinecone initialization code 
# 'from pinecone import Pinecone, ServerlessSpec
# import time
# 
# pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
# 
# cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
# region = os.environ.get('PINECONE_REGION') or 'us-east-1'
# spec = ServerlessSpec(cloud=cloud, region=region)
# 
# index_name = PINECONE_INDEX_NAME
# 
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=embeddings.dimension,
#         metric="cosine",
#         spec=spec
#     )
# index = pc.Index(index_name)
# for i, chunk in enumerate(chunks):
#     vector = embedding.embed_query(chunk.page_content)
#     index.upsert([
#         {
#             "id": f"ccmt-chunk-{i}",
#             "values": vector,
#             "metadata": {"text": chunk.page_content}
#         }
#     ])
# 
# 
# 
# 
# namespace = "wondervector5000"
# 
# vectordb = PineconeVectorStore.from_documents(
#     documents=chunks,
#     index_name=index_name,
#     embedding=embedding,
#     namespace=namespace
#    
# )
# 
# print("Index before upsert:")
# print(pc.Index(index_name).describe_index_stats())
# print("\n") '

# In[ ]:





# In[56]:


namespace = "wondervector5000"

vectordb = PineconeVectorStore.from_existing_index(
    #documents=chunks,
    index_name=index_name,
    embedding=embedding,
    namespace=namespace

)


# In[57]:


retriever = vectordb.as_retriever(search_kwargs={"k": 5})


# In[ ]:





# ## rag chain 

# In[58]:


PROMPT_TEMPLATE = """
The user is a candidate
You are a helpful CCMT counseling assistant.
Always provide complete answers.
Never reply with vague phrases like "refer to the table" or "see section".
Include exact fee amounts, names, and explanations when available.
Answer the question based only on the following context:
{context}
Answer the question based on the above context: {question}.
Provide a detailed answer.
Don’t justify your answers.
Do not say "according to the context" or "mentioned in the context" or similar.
"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


# In[59]:


chat_model=ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY , model='gemini-2.0-flash-exp')
parser = StrOutputParser()


# ### retriever: A vector retriever (e.g., FAISS or Chroma) — returns the most relevant k documents based on similarity to the question.
# ### format_docs: A function to convert those document objects into a single formatted string (so it can go into the prompt).

# In[60]:


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# In[61]:


rag_chain = {"context": retriever | format_docs , "question": RunnablePassthrough()} | prompt_template | chat_model | parser


# ## Streamlit UI
# 

# In[64]:


# Set page config at the very beginning
st.set_page_config(page_title="CCMT Chatbot", layout="wide")

# ----------- Unique Visit Logging Logic -------------
COUNT_FILE = "user_count.txt"

# Create the file if it doesn't exist
if not os.path.exists(COUNT_FILE):
    with open(COUNT_FILE, "w") as f:
        f.write("")

# Only add one star per session
if "counted" not in st.session_state:
    with open(COUNT_FILE, "a") as f:
        f.write("*")
    st.session_state.counted = True

# Read total user count
with open(COUNT_FILE, "r") as f:
    stars = f.read()
    total_users = stars.count("*")

# ----------- Main Chatbot UI ------------------------


st.markdown(
    """
   <style>
/* App background and text */
.stApp {
    background: linear-gradient(135deg, #141e30, #243b55);  /* deep dark gradient */
    color: white !important;
}

/* Title and all markdown text */
h1, h2, h3, h4, h5, h6, p, span, div {
    color: white !important;
}

/* Input box style */
.stTextInput > div > div > input {
    background-color: #2c3e50;
    color: white;
    border: 1px solid #ffffff33;
    border-radius: 10px;
    padding: 0.5em;
}

/* Button style */
.stButton > button {
    background-color: #00cec9;
    color: black;
    border: none;
    border-radius: 10px;
    padding: 0.5em 1.2em;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #81ecec;
    color: black;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #1f1f1f;
    color: white;
}

section[data-testid="stSidebar"] p {
    color: white;
}
    """,
    unsafe_allow_html=True
)

st.title("🎓 CCMT Counselling Chatbot")
st.markdown("Ask anything about CCMT rules, rounds, fees, etc...")

# Sidebar content
st.sidebar.markdown(f"🌟 **Unique Visits:** {total_users}")

# Input section
query = st.text_input("💬 Ask your question:")

# Button and response
if st.button("Get Answer"):
    if query.strip():
        with st.spinner("Thinking..."):
            result = rag_chain.invoke(query)
            st.markdown("**Answer:**")
            st.write(result)
    else:
        st.warning("Please enter a question first.")



# In[ ]:





# In[ ]:





# In[ ]:




