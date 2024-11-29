# %% [markdown]
# Importamos las dependencias necesarias

# %%
import streamlit as st
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# %% [markdown]
# Instanciamos el llm, el modelo de embedings y la base de datos vectorial

# %%

GOOGLE_API_KEY = 'AIzaSyBq0R6YpJn5oW96RyFpxzVKjWVw0TvsiEs'
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=GOOGLE_API_KEY, temperature=0)
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
vector_store_connection = SKLearnVectorStore(embedding=embedding_function,
                                            persist_path="./db/Db_web-plip_split",
                                            serializer="parquet")

# %% [markdown]
# Instanciamos el retriver que vamos a usar para buscar en la base de datos 

# %%


# %% [markdown]
# Creamos el template con una funcion para que pueda personalizarse

# %%

retriever = vector_store_connection.as_retriever()

# %% [markdown]
# Creamos la cadena para hacer preguntas

# %%
# Creamos la funcion para formatear la respuesta

def generate_response(response):
    
    paginas = ""
    for doc in response['context']:
        paginas = f"{paginas}\n{doc.metadata['source']}"
    if response['answer'] == 'No tengo información al respecto.\n':
        respuesta = f"{response['answer']}\n\nRecuerda que solo puedo responder preguntas sobre plip"
    else:
        respuesta = f"{response['answer']}"
    
    return respuesta

# %%
st.set_page_config(
    page_title='Chat bot plip'
)

# %%
st.title('Chat bot PliP')

# Asignamos la personalidad al bot dependiendo del valor que tenga el boton
personality = st.text_input(
    '¿Qué personalidad deberia tener el bot?',
    value='Eres una persona de soporte que recibe preguntas de los usuarios y eres muy animado y atento.'
)

bot_description  = f"""Responde siempre en español.
{personality}
si la respuesta no la puedes dar con el contexto debes responder: 'No tengo información al respecto'.
Siempre conesta como si el conocimiento fuera tu conocimiento.
y solo basandote en el siguiente contexto: {{context}}

"""


#Creamos la cadena para responder preguntas


#Creamos el espacio para que el usuario ponga su pregunta


if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

#Creamos el prompt, con todo lo que hemos establecido

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", bot_description),
        MessagesPlaceholder('chat_history'),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

chain = create_retrieval_chain(retriever, question_answer_chain)

query_text = st.text_input(
    "¿Qué deseas saber?",
    placeholder="Escribe tu pregunta aquí",
    key='user_input'
)

# %% [markdown]
# Se crea un formulario que se limpiará cuando se envíe la pregunta
#  

# %%

with st.form(
    "myform",
    clear_on_submit=True
):
    
    submitted = st.form_submit_button(
    "Enviar",
    disabled=not (query_text)
    )
    if submitted:
        with st.spinner(
            "Escribiendo..."
            ):
            
            response = chain.invoke({'input':query_text,
                                    'personality':personality,
                                    'chat_history': st.session_state['chat_history']})
            response = generate_response(response)
            st.session_state['chat_history'].append(HumanMessage(query_text))
            st.session_state['chat_history'].append(AIMessage(response))

chat_display = ''
for msg in st.session_state['chat_history']:
    if isinstance(msg, HumanMessage):
        chat_display += f'Humano: {msg.content}\n\n'
    elif isinstance(msg, AIMessage):
        chat_display += f'Bot: {msg.content}\n\n'
query_text.value = ''
st.text_area('Chat', value=chat_display, height=400, key='chat_area')





