from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI




def format_docs(docs):
    return "\n\n---\n\n".join(
        doc.page_content for doc in docs
    )


def create_rag_chain(retriever,provider,api_key):
    system_prompt = """
    You are a helpful assistant.
    Answer the question using ONLY the context provided below.
    If the context does not contain enough information, say so clearly.
    Also don't answer long!

    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    if provider == "groq":
        llm = ChatGroq(
        api_key=api_key,
        model="qwen/qwen3-32b",
        temperature=0.3
    )

    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model="gemini-2.5-flash",
            temperature=0.3
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain