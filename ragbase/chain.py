import re
from operator import itemgetter
from typing import List

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever

from ragbase.config import Config
from ragbase.session_history import get_session_history

SYSTEM_PROMPT = """
Utilize the provided contextual information to respond to the user question.
If the answer is not found within the context, state that the answer cannot be found.
Prioritize concise responses (maximum of 3 sentences) and use a list where applicable.
The contextual information is organized with the most relevant source appearing first.
Each source is separated by a horizontal rule (---).

Context:
{context}

Use markdown formatting where appropriate.
Please give confidence score for each answer.
"""


def remove_links(text: str) -> str:
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", text)


def format_documents(documents: List[Document]) -> str:
    texts = []
    for doc in documents:
        texts.append(doc.page_content)
        texts.append("---")

    return remove_links("\n".join(texts))

def input_guardrail(user_input):
    prohibited_words = ["idiot", "stupid", "dumb"]
    
    if any(word in user_input['question'].lower() for word in prohibited_words):
        # Add a special flag to indicate prohibited content
        user_input["prohibited"] = True
        user_input["warning"] = "Warning: Your input contains words that may be inappropriate. Please revise your question."
    
    return user_input

def create_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Runnable:
    from langchain_core.runnables.base import RunnableLambda

    # Apply input guardrail
    validate = RunnableLambda(input_guardrail)
    
    # Define a branch function that checks for prohibited content
    def branch_on_prohibited(input_dict):
        if input_dict.get("prohibited", False):
            # If prohibited, return the warning directly
            return input_dict["warning"]
        
        # Otherwise, continue with the normal chain
        return (
            RunnablePassthrough.assign(
                context=itemgetter("question") 
                | retriever.with_config({"run_name": "context_retriever"})
                | format_documents
            )
            | prompt
            | llm
        ).invoke(input_dict)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    chain = validate | RunnableLambda(branch_on_prohibited)

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    ).with_config({"run_name": "chain_answer"})


async def ask_question(chain: Runnable, question: str, session_id: str):
    async for event in chain.astream_events(
        {"question": question},
        config={
            "callbacks": [ConsoleCallbackHandler()] if Config.DEBUG else [],
            "configurable": {"session_id": session_id},
        },
        version="v2",
        include_names=["context_retriever", "chain_answer"],
    ):
        event_type = event["event"]
        if event_type == "on_retriever_end":
            yield event["data"]["output"]
        if event_type == "on_chain_stream":
            # Check if the chunk is a string (warning message) or an object with content
            if isinstance(event["data"]["chunk"], str):
                yield event["data"]["chunk"]
            else:
                yield event["data"]["chunk"].content