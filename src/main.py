import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

# Configuração da Chave de API (Simulação - Na prática usaria .env)
# os.environ["OPENAI_API_KEY"] = "SUA_CHAVE_AQUI"

def main():
    print("--- Iniciando Processo de RAG (Retrieval-Augmented Generation) ---")
    
    # 1. Carregamento de Dados (Simulando um documento da empresa)
    # Na prática, leríamos um arquivo da pasta 'data/'
    texto_base = """
    A Future AI Labs é uma empresa líder em inovação.
    Nossa missão é transformar negócios com Inteligência Artificial.
    Estamos contratando Engenheiros de IA focados em LLMs e Agentes.
    O contato para envio de currículos é daniel.reis@futureai.consulting.
    Nossa stack principal envolve Python, LangChain e PyTorch.
    """
    
    # Criando um arquivo temporário para simular o carregamento
    with open("data/info_empresa.txt", "w", encoding="utf-8") as f:
        f.write(texto_base)
    
    loader = TextLoader("data/info_empresa.txt", encoding="utf-8")
    documents = loader.load()
    
    # 2. Chunking (Divisão do texto em pedaços menores)
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    print(f"Documento dividido em {len(texts)} pedaços.")

    # 3. Embeddings e Vector Store (Transformar texto em números e salvar)
    # Observação: Sem uma chave de API real, esta parte vai dar erro ao rodar.
    # Mas o código está estruturalmente pronto para o GitHub.
    try:
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        
        # 4. Criação da Chain de QA (Pergunta e Resposta)
        qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())
        
        query = "Quais tecnologias a empresa usa?"
        print(f"\nPergunta: {query}")
        
        resposta = qa.run(query)
        print(f"Resposta da IA: {resposta}")
        
    except Exception as e:
        print("\n[Aviso] Para rodar a inferência real, configure a OPENAI_API_KEY.")
        print("O código está pronto para demonstrar a estrutura de RAG.")

if __name__ == "__main__":
    main()