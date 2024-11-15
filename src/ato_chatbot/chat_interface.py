# %%
import json
import logging
import os

import streamlit as st
from llama_index.core import (PromptTemplate, Settings, StorageContext,
                              VectorStoreIndex)
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from openai import OpenAI
from qdrant_client import QdrantClient

# %%

# config
QDRANT_URI = st.secrets["QDRANT_URI"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
QDRANT_COLLECTION_NAME = st.secrets["QDRANT_COLLECTION_NAME"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
LLM_MODEL = st.secrets["LLM_MODEL"]
TOP_K = 7

# %%

# create logger, setting basic formatting
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# %%
# prompt template

SYSTEM_PROMPT = """
You are an expert assistant specializing in Australian Taxation Office (ATO) matters. Your primary role is to provide accurate, factual information about Australian taxation by carefully analyzing and synthesizing the provided context.

Instructions for Response:
1. Focus ONLY on Australian taxation and ATO-related matters
2. Analyze all provided context thoroughly before answering
3. Only use information explicitly present in the context
4. Cite specific sections or sources when making statements
5. If information is partial or unclear, acknowledge the limitations
6. For non-tax related financial queries, politely redirect users to seek appropriate professional advice

Acceptable Topics:
- Australian tax laws and regulations
- ATO services and requirements
- Tax deductions and returns
- GST and income tax matters
- Business tax obligations in Australia
- Taxation aspects of superannuation
- Tax-related financial reporting requirements

---
**Response Structure:**

[Provide a clear, context-based answer that directly addresses the user's tax-related question]

**Key Information:**
- **Verified Facts:** [List key tax-related facts from the context with source references]
- **Requirements:** [List specific tax requirements mentioned in the context]
- **Important Dates/Deadlines:** [List relevant tax dates from the context]

**Source References:**
[List relevant ATO source URLs from the context, with brief descriptions of what information was drawn from each]

---
**Confidence Level:**
- High: All tax-related information directly supported by context
- Partial: Some aspects need additional verification
- Limited: Insufficient context to fully answer

*Note: This response is based solely on official ATO documentation. For complex taxation matters or non-tax financial advice, please consult appropriate professionals.*
"""

USER_PROMPT_TEMPLATE = """
**User Question:** {query_str}
**Context (JSON):** {context_str}
"""

# %%


@st.cache_resource
def initialize_llm(api_key: str) -> OpenAI:
    """Initialize the OpenAI language model."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    return client


@st.cache_resource
def initialize_llama_index_llm(api_key: str) -> LlamaIndexOpenAI:
    """Initialize the LlamaIndex OpenAI language model."""
    llm = LlamaIndexOpenAI(api_key=OPENAI_API_KEY)
    return llm


@st.cache_resource
def initialize_index() -> VectorStoreIndex:
    """Create a VectorStoreIndex from the documents."""

    client = QdrantClient(url=QDRANT_URI, api_key=QDRANT_API_KEY)
    vector_store = QdrantVectorStore(
        client=client, collection_name=QDRANT_COLLECTION_NAME
    )

    return VectorStoreIndex.from_vector_store(vector_store)


@st.cache_resource
def initialize_prompt() -> PromptTemplate:
    """Initialize the custom prompt template."""
    return PromptTemplate(USER_PROMPT_TEMPLATE)


def rephrase_query_function(client: OpenAI, query: str) -> str:
    """Rephrase the user query to be more accurate and searchable."""

    rephrase_prompt = """Rephrase the following question to be more accurate and searchable, 
    focusing on key taxation terms and concepts"""

    rephrased = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": rephrase_prompt},
            {"role": "user", "content": query},
        ],
    )

    return rephrased.choices[0].message.content


def intetion_recognition_function(client: OpenAI, query: str) -> bool:
    """Recognize the user's intention based on the query."""

    try:
        intention_prompt = """
        Identify if the query is related to Australian taxation or the ATO (Australian Taxation Office).
        
        Return True ONLY if the query is about:
        - Australian tax laws, regulations, or procedures
        - ATO services or requirements
        - Tax deductions, returns, or obligations
        - GST, income tax, or other Australian tax types
        - Business tax obligations in Australia
        - Australian financial matters directly related to taxation
        
        Return False for:
        - Non-tax related questions
        - General chat or greetings
        - Questions about non-Australian tax systems
        - Personal or off-topic queries
        - General financial advice not related to taxation
        
        Return response in JSON format: {"intention": true/false}
        """

        intention = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": intention_prompt},
                {"role": "user", "content": query},
            ],
        )
        response_text = intention.choices[0].message.content
        logger.debug(f"Intention recognition response: {response_text}")

        intention: dict = json.loads(response_text)
        logger.debug(f"Parsed intention response: {intention}")

        return intention["intention"]
    except Exception as e:
        logger.error(
            f"Error recognizing intention: {e}, api returned: {intention.choices[0].message.content}"
        )
        return False


def retrieve_relevant_nodes(query: str, index: VectorStoreIndex) -> list:
    """Retrieve relevant nodes from the index based on the query."""
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    nodes = retriever.retrieve(query)
    return nodes


if __name__ == "__main__":
    # Initialize the language model
    llm = initialize_llama_index_llm(OPENAI_API_KEY)
    client = initialize_llm(OPENAI_API_KEY)

    # Update Settings instead of ServiceContext
    Settings.llm = llm

    # Create the index
    index = initialize_index()

    # Add introduction/header
    st.title("Welcome to the ATO Tax Information Assistant 🧾")
    st.markdown(
        """
    Hello! 👋 I'm your ATO Tax Information Assistant, here to provide you with accurate and up-to-date information on Australian taxation matters. 
    All insights are sourced directly from the [Australian Taxation Office website](https://www.ato.gov.au/).

    **Important Information:**
    - **Source:** All data is derived from official ATO documentation 📚
    - **Purpose:** This tool is designed for general informational purposes and should not replace professional tax advice ⚠️
    - **Consultation:** For personalized tax advice, please reach out to a certified tax professional 👨‍💼👩‍💼
    - **Disclaimer:** This is a demonstration project showcasing RAG (Retrieval-Augmented Generation) capabilities. The ATO website has not been fully scraped, so responses may not be comprehensive.

    **Example Questions You Can Ask:**
    - "What are the GST requirements for small businesses?"
    - "How do I calculate capital gains tax on an investment property?"
    - "What superannuation contributions can I claim as a tax deduction?"

    **About the Author:**
    - **Developer:** Ted Zhao
    - **LinkedIn:** [Ted Zhao](https://www.linkedin.com/in/ted-zhao/)
    - **Email:** ted.zhao.au@gmail.com
    - **Resume:** [GitHub Resume](https://github.com/tade0726/tedzhao-resume)

    💼 *Currently open to new opportunities! Feel free to contact me to discuss:*
    - RAG applications and implementations
    - Professional opportunities
    - AI/ML collaborations

    Let's get started with your queries! 🚀
    """
    )

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = LLM_MODEL

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Your message"):
        logger.debug(f"Query: {query}")

        # Check query intention
        if not (intention := intetion_recognition_function(client, query)):
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                message = """I apologize, but I can only answer questions related to Australian taxation and ATO matters. 

Please rephrase your question to focus on:
- Australian tax laws and regulations
- ATO services and requirements
- Tax deductions and returns
- GST and income tax
- Business tax obligations in Australia
- Taxation aspects of superannuation
- Tax-related financial matters

For non-tax related financial advice, please consult a financial advisor. How can I help you with your Australian tax-related query?"""
                st.warning(message)

            # Add the exchange to session state
            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": message})
        else:
            # 1. Rephrase the query for better search
            rephrase_query: str = rephrase_query_function(client, query)

            logger.debug(f"Rephrased query: {rephrase_query}")

            with st.chat_message("user"):
                st.markdown(query)

            # 2. Retrieve relevant nodes
            relevant_nodes = retrieve_relevant_nodes(rephrase_query, index)

            # 3. Generate final response
            context = [
                {"source": node.metadata["source"], "content": node.text}
                for node in relevant_nodes
            ]

            logger.debug(f"Context: {context}")

            # Get the prompt template
            prompt_template = initialize_prompt()

            # Format the prompt with context and rephrase_query
            formatted_prompt = prompt_template.format(
                context_str=json.dumps(context, indent=2), query_str=rephrase_query
            )

            logger.debug(f"Formatted prompt: {formatted_prompt}")

            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    st.session_state.messages.append(
                        {"role": "user", "content": rephrase_query}
                    )

                    stream = client.chat.completions.create(
                        model=st.session_state["openai_model"],
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                        + [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ]
                        + [{"role": "user", "content": formatted_prompt}],
                        stream=True,
                    )

                response = st.write_stream(stream)

            st.session_state.messages.append({"role": "assistant", "content": response})


# %%
