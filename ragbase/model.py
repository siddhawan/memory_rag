
from langchain_community.document_compressors.flashrank_rerank import \
    FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from ragbase.config import Config


def create_llm() -> BaseLanguageModel:
    if Config.Model.USE_LOCAL:
        return ChatOllama(
            model=Config.Model.LOCAL_LLM,
            temperature=Config.Model.TEMPERATURE,
            keep_alive="1h",
            max_tokens=Config.Model.MAX_TOKENS,
        )
    else:
        return ChatGroq(
            temperature=Config.Model.TEMPERATURE,
            model_name=Config.Model.REMOTE_LLM,
            max_tokens=Config.Model.MAX_TOKENS,
            groq_api_key='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'
        )


def create_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=Config.Model.EMBEDDINGS)


def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(model=Config.Model.RERANKER)
