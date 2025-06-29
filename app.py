# Correção para o SQLite no Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import google.generativeai as genai
import os
import pypdf
import requests
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Constantes de configuração
MAX_FILE_SIZE_MB = 5
MAX_OUTPUT_TOKENS = 1024

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

def log_to_discord(title, fields, mention_user=False):
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL") or st.secrets.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("AVISO: URL do Webhook do Discord não configurada. O log será pulado.")
        return

    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    embed = {
        "title": title,
        "color": 5814783 if "Requisição" in title else 3066993, # Azul para requisição, verde para upload
        "fields": fields,
        "footer": {
            "text": f"Bot de Logs - Assistente de Documentos • {timestamp}"
        }
    }

    content = ""
    if mention_user:
        user_id = os.getenv("DISCORD_USER_ID_TO_MENTION") or st.secrets.get("DISCORD_USER_ID_TO_MENTION")
        if user_id:
            content = f"<@{user_id}>"

    payload = {
        "content": content,
        "embeds": [embed]
    }

    try:
        requests.post(webhook_url, json=payload)
    except Exception as e:
        print(f"ERRO: Falha ao enviar log para o Discord: {e}")

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
            request_id = str(uuid.uuid4())
            log_fields = [
                {"name": "Pergunta do Usuário", "value": f"```{user_question}```"},
                {"name": "ID da Requisição", "value": f"`{request_id}`", "inline": True}
            ]
            log_to_discord("Nova Requisição de Usuário", log_fields, mention_user=True)

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

            model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash-latest",
                temperature=0.3,
                max_output_tokens=MAX_OUTPUT_TOKENS
            )

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
    st.markdown(f"Envie seus documentos PDF (limite de {MAX_FILE_SIZE_MB} MB por arquivo) e clique em 'Processar'.")

    pdf_docs_uploaded = st.file_uploader(
        "Carregar Arquivos PDF", accept_multiple_files=True, type="pdf"
    )

    if st.button("Processar Documentos", type="primary", use_container_width=True):
        if pdf_docs_uploaded:
            valid_docs = []
            log_fields = []
            for doc in pdf_docs_uploaded:
                file_size_mb = doc.size / (1024 * 1024)
                if file_size_mb > MAX_FILE_SIZE_MB:
                    st.warning(f"O arquivo '{doc.name}' ({file_size_mb:.2f} MB) excede o limite de {MAX_FILE_SIZE_MB} MB e será ignorado.")
                    continue
                valid_docs.append(doc)
                log_fields.append({
                    "name": f"📄 {doc.name}",
                    "value": f"`{file_size_mb:.2f} MB`",
                    "inline": True
                })

            if not valid_docs:
                st.error("Nenhum dos arquivos enviados é válido. Por favor, envie arquivos dentro do limite de tamanho.")
            else:
                if log_fields:
                    log_to_discord("Upload de Documentos Recebido", log_fields, mention_user=True)

                with st.spinner("Processando... Isso pode levar um momento."):
                    raw_text = get_pdf_text(valid_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        vector_store = get_vector_store(text_chunks)
                        st.session_state.vector_store = vector_store
                        if vector_store:
                            st.session_state.processed = True
                            st.session_state.chat_history = []
                            st.success("Documentos processados com sucesso! Agora você pode fazer perguntas.")
                    else:
                        st.error("Não foi possível extrair texto dos PDFs válidos. Por favor, verifique os arquivos.")
        else:
            st.warning("Por favor, carregue pelo menos um arquivo PDF.")

    st.markdown("---")
    if st.button("Limpar e Reiniciar", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.processed = False
        st.cache_resource.clear()
        st.rerun()

    st.markdown(f"<div style='text-align: center; font-size: 0.8em; color: grey;'>© {datetime.now().year} Assistente de Documentos</div>", unsafe_allow_html=True)

st.title("🤖 Assistente de Conversação")
st.markdown("Bem-vindo(a)! Eu posso responder a perguntas sobre os documentos que você enviou.")

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
