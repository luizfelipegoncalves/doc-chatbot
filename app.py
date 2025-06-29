import streamlit as st
import google.generativeai as genai
import os
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

st.set_page_config(
    page_title="Assistente de Documentos",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Chave de API do Google não encontrada. Por favor, configure-a nos segredos do Streamlit (secrets) ou em um arquivo .env.")
        st.stop()
    genai.configure(api_key=api_key)
except Exception as e:
    st.error(f"Erro ao configurar a API do Google: {e}")
    st.stop()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = pypdf.PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.warning(f"Não foi possível ler o arquivo {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource(show_spinner="Criando índice dos documentos...")
def get_vector_store(_text_chunks):
    if not _text_chunks:
        st.error("Nenhum texto foi extraído dos PDFs. Não é possível criar o índice.")
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = Chroma.from_texts(_text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Erro ao criar o índice de vetores: {e}")
        return None

def handle_user_input(user_question):
    if 'vector_store' not in st.session_state or st.session_state.vector_store is None:
        st.error("O índice de documentos não foi inicializado. Por favor, envie e processe os documentos primeiro.")
        return

    with st.spinner("Buscando respostas..."):
        try:
            prompt_template = """
            Você é um assistente prestativo para tarefas de perguntas e respostas.
            Responda à pergunta da forma mais detalhada possível, baseando-se SOMENTE no contexto fornecido.
            Se a resposta não estiver no contexto fornecido, apenas diga: "A resposta não está disponível nos documentos fornecidos."
            Não forneça respostas baseadas em seu próprio conhecimento.

            Contexto:
            {context}

            Pergunta:
            {question}

            Resposta:
            """

            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)

            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

            qa_chain = RetrievalQA.from_chain_type(
                llm=model,
                chain_type="stuff",
                retriever=st.session_state.vector_store.as_retriever(),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=False
            )

            response = qa_chain.invoke({"query": user_question})

            st.session_state.chat_history.append({"role": "assistant", "content": response["result"]})

        except Exception as e:
            error_message = f"Ocorreu um erro: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed" not in st.session_state:
    st.session_state.processed = False

with st.sidebar:
    st.title("📄 Central de Documentos")
    st.markdown("Envie seus documentos PDF e clique em 'Processar' para começar.")

    pdf_docs = st.file_uploader(
        "Carregar Arquivos PDF", accept_multiple_files=True, type="pdf",
        help="Você pode carregar um ou mais arquivos PDF."
    )

    if st.button("Processar Documentos", type="primary", use_container_width=True):
        if pdf_docs:
            with st.spinner("Processando... Isso pode levar um momento."):
                raw_text = get_pdf_text(pdf_docs)

                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_vector_store(text_chunks)
                    st.session_state.vector_store = vector_store

                    if vector_store:
                        st.session_state.processed = True
                        st.session_state.chat_history = []
                        st.success("Documentos processados com sucesso! Agora você pode fazer perguntas.")
                else:
                    st.error("Não foi possível extrair texto do(s) PDF(s) enviado(s). Por favor, verifique os arquivos.")
        else:
            st.warning("Por favor, carregue pelo menos um arquivo PDF.")

    st.markdown("---")
    if st.button("Limpar e Reiniciar", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.processed = False
        st.cache_resource.clear()
        st.rerun()

    st.markdown(f"<div style='text-align: center; font-size: 0.8em; color: grey;'>© {datetime.now().year} Assistente de Documentos Gemini</div>", unsafe_allow_html=True)

st.title("🤖 Assistente de Conversação Gemini")
st.markdown("Bem-vindo! Eu posso responder a perguntas sobre os documentos que você enviou.")

if "chat_history" in st.session_state:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if st.session_state.processed:
    if user_question := st.chat_input("Faça uma pergunta sobre seus documentos..."):
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        handle_user_input(user_question)
        st.rerun()
else:
    st.info("Por favor, envie e processe seus documentos PDF na barra lateral para iniciar a conversa.")
