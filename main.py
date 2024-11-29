# %% [markdown]
# Importamos las dependencias necesarias

# %%
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

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

template = """Responde seimpre en español.
Eres una persona de soporte que recibe preguntas de los usuarios y eres muy animado y atento.
si la respuesta no la puedes dar con el contexto debes responder: 'No tengo información al respecto'.
Siempre conesta como si el conocimiento fuera tu conocimiento.
y solo basandote en el siguiente contexto:

{context}

Pregunta: {question}
"""
prompt = PromptTemplate.from_template(template)

# %% [markdown]
# Creamos la cadena para hacer preguntas

# %%
QA_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store_connection.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True 
)


#QA_chain.invoke(query)

def generate_response(chain, query):
    response = chain.invoke(query)
    paginas = ""
    for doc in response['source_documents']:
        paginas = f"{paginas}\n{doc.metadata['source']}"
    if response['result'] == 'No tengo información al respecto.':
        respuesta = f"response['result']\n\n Recuerda que solo puedo responder preguntas sobre plip"
    else:
        respuesta = f"{response['result']} \nInformación obtenida de: \n{paginas}"
    return respuesta

# %%
st.set_page_config(
    page_title='Chat bot plip'
)

# %%
st.title('Chat bot plip')

# %%
query_text = st.text_input(
    "¿Qué deseas saber?",
    placeholder="Escribe tu pregunta aquí"
)

# %% [markdown]
# Se crea un formulario que se limpiará cuando se envíe la pregunta
#  

# %%
result = []
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
            
            
            response = generate_response(QA_chain, query_text)

            result.append(response)
if len(result):
    st.info(result)




