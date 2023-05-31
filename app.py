import os
import uuid
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from memory import EmbeddingMemory
from prompt_template import PaimonPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title='MemoryBot🤖', layout='centered', initial_sidebar_state='collapsed')
# Initialize session states
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())
if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""
# Set up the Streamlit app layout
st.title("🤖 PAIMON")

@st.cache_data
def get_system_prompt_text():
    with open('public/system.txt', 'r') as f:
        system_text = f.read()
    return system_text

def get_prompt():
    system_prompt = PaimonPromptTemplate(input_variables=["input", "history"])
    return system_prompt
@st.cache_resource
def init_embedding_memory():
    memory = EmbeddingMemory(name='with-paimon', top_k=3)
    return memory

@st.cache_resource 
def load_chat_chain(openai_api_key, model_name, openai_proxy=os.getenv('https_proxy')):
    llm = ChatOpenAI(temperature=1.0, model_name=model_name, openai_api_key=openai_api_key, openai_proxy=openai_proxy)
    chat_chain = LLMChain(llm=llm, prompt=get_prompt(), verbose=True)
    return chat_chain

# Set up sidebar with various options, init chat conversation instance
with st.sidebar:
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo'])
    API_KEY = st.text_input("API-KEY", os.getenv('openai_api_key'), type="password")
# Session state storage would be ideal
if API_KEY:
    chat_chain = load_chat_chain(openai_api_key=API_KEY, model_name=MODEL)
else:
    st.sidebar.warning('API key required to try this app.The API key is not stored in any form.')

embedding_memory = init_embedding_memory()
def get_embedding_memory_messages(user_input):
    documents = embedding_memory.query(user_input)
    messages = []
    print('memory:')
    print(documents)
    for document in documents:
        human, ai = document.split('\n')
        messages.append(human)
        messages.append(ai)
    return messages

# Conversion PAGE
def submit():
    st.session_state.user_input = st.session_state.input
    st.session_state.input = ''
chat_list = st.container()
chat_list_expander = chat_list.expander("Conversation", expanded=True)
chat_list.text_input("You: ", key="input", on_change=submit, placeholder="Your AI assistant here! Ask me anything ...", label_visibility='hidden')
# Get the user input
user_input = st.session_state.user_input
# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
if user_input:
    memory_messages = get_embedding_memory_messages(user_input)
    if(len(memory_messages) > 0):
        print("Memory messages: ", memory_messages)
    output = chat_chain.run(input=user_input, history=memory_messages)
    st.session_state.past.append(user_input)  
    st.session_state.generated.append(output)
    document = f"旅行者：{user_input}\n{output}"
    embedding_memory.add([document])

# Conversation list
with chat_list_expander:
    for i in range(len(st.session_state['generated'])):
        st.info(st.session_state["past"][i], icon="🧐")
        st.success(st.session_state["generated"][i], icon="🤖")