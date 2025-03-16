
from langchain_community.document_compressors.flashrank_rerank import \
    FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_core.language_models import BaseLanguageModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ragbase.config import Config


def create_llm() -> BaseLanguageModel:
    if Config.Model.USE_LOCAL:
            model_name = Config.Model.LOCAL_LLM  # Ensure this is a Hugging Face model name
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

            text_gen_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                temperature=Config.Model.TEMPERATURE,
                max_new_tokens=10
            )

            return HuggingFacePipeline(pipeline=text_gen_pipeline)

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
