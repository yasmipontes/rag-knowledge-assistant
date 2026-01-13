# 🧠 RAG Knowledge Assistant

Este projeto implementa um pipeline de **RAG (Retrieval-Augmented Generation)** para sistemas de perguntas e respostas baseados em documentos privados. O objetivo é demonstrar a aplicação de **LLMs (Large Language Models)**, **Busca Semântica** e **Engenharia de Prompts** para resolver problemas de negócio, garantindo que a IA responda com base em um contexto específico e não apenas em seu treinamento prévio.

## 🚀 Tecnologias Utilizadas

O projeto foi construído utilizando stack moderna de Inteligência Artificial focada em performance e escalabilidade:

* **Python 3.10+**: Linguagem base.
* **LangChain**: Framework para orquestração de fluxos de IA e manipulação de cadeias de pensamento.
* **FAISS (Facebook AI Similarity Search)**: Banco de dados vetorial para busca eficiente de similaridade (Embeddings).
* **OpenAI API**: Utilizada para geração de Embeddings e modelo de Chat (LLM).
* **TikToken**: Tokenização eficiente para controle de janelas de contexto.

## ⚙️ Arquitetura da Solução

O fluxo de trabalho do sistema segue as etapas de um RAG clássico:

1.  **Ingestão de Dados**: Carregamento de documentos textuais (txt, pdf, md).
2.  **Chunking**: Divisão do texto em fragmentos menores (chunks) para otimizar a recuperação.
3.  **Embedding**: Conversão dos textos em vetores numéricos de alta dimensão.
4.  **Vector Store**: Armazenamento dos vetores utilizando FAISS.
5.  **Retrieval (Recuperação)**: Busca semântica dos trechos mais relevantes à pergunta do usuário.
6.  **Generation (Geração)**: Envio do contexto recuperado + pergunta para o LLM gerar a resposta final.

## 📂 Estrutura do Projeto

```bash
rag-knowledge-assistant/
├── data/              # Diretório para armazenamento de documentos fonte
├── src/
│   └── main.py        # Código principal do pipeline RAG
├── requirements.txt   # Dependências do projeto
└── README.md          # Documentação