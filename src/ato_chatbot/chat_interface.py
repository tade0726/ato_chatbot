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
You are an expert assistant specializing in Australian Taxation Office (ATO) matters. Your primary role is to provide accurate, factual information by carefully analyzing and synthesizing the provided context.

Instructions for Response:
1. Analyze all provided context thoroughly before answering
2. Only use information explicitly present in the context
3. Cite specific sections or sources when making statements
4. If information is partial or unclear, acknowledge the limitations

---
**Response Structure:**

[Provide a clear, context-based answer that directly addresses the user's question]

**Key Information:**
- **Verified Facts:** [List key facts from the context with source references]
- **Requirements:** [List specific requirements mentioned in the context]
- **Important Dates/Deadlines:** [List relevant dates from the context]

**Source References:**
[List relevant source URLs from the context, with brief descriptions of what information was drawn from each]

---
**Confidence Level:**
- High: All information directly supported by context
- Partial: Some aspects need additional verification
- Limited: Insufficient context to fully answer

*Note: This response is based solely on official ATO documentation. For complex situations, please consult a tax professional.*
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
def initialize_index(_llm: OpenAI) -> VectorStoreIndex:
    """Create a VectorStoreIndex from the documents."""

    # Update Settings instead of ServiceContext
    Settings.llm = _llm
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


def retrieve_relevant_nodes(query: str, index: VectorStoreIndex) -> list:
    """Retrieve relevant nodes from the index based on the query."""
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    nodes = retriever.retrieve(query)
    return nodes


if __name__ == "__main__":
    # Initialize the language model
    llm = initialize_llama_index_llm(OPENAI_API_KEY)
    client = initialize_llm(OPENAI_API_KEY)

    # Create the index
    index = initialize_index(llm)

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
