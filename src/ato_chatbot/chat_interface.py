# %%
import json
import logging
import os

import streamlit as st
from llama_index.core import (PromptTemplate, ServiceContext, StorageContext,
                              VectorStoreIndex)
from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from openai import OpenAI
from qdrant_client import QdrantClient

# %%

# config
QDRANT_URI = "http://127.0.0.1:6333"
COLLECTION_NAME = "ato-chatbot-simple-index-32"
LLM_MODEL = "gpt-4o"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# %%

# create logger, setting basic formatting
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# %%
# prompt template

CUSTOM_PROMPT_TEMPLATE = """
You are an expert assistant specializing in Australian Taxation Office (ATO) matters. Use the provided context to answer the user's question. If the context lacks the necessary information, state this clearly.

**Context (JSON):** {context_str}

**User Question:** {query_str}

---

**Response:**

[Provide a clear and concise answer to the user's question.]

**Key Information:**

- **Main Points:** [Summarize key points.]
- **Requirements:** [List any requirements.]
- **Important Dates:** [List relevant dates.]

**Related Resources:**

- [Topic]: [Source URL]
- [Topic]: [Source URL]

---

*Note: If the context does not contain the required information, please indicate this clearly. All information is sourced from official ATO documentation.*
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
def initialize_index(
    qdrant_uri: str, collection_name: str, _llm: OpenAI
) -> VectorStoreIndex:
    """Create a VectorStoreIndex from the documents."""

    service_context = ServiceContext.from_defaults(llm=_llm)
    client = QdrantClient(url=qdrant_uri)
    vector_store = QdrantVectorStore(client=client, collection_name=collection_name)

    return VectorStoreIndex.from_vector_store(
        vector_store, service_context=service_context
    )


@st.cache_resource
def initialize_prompt() -> PromptTemplate:
    """Initialize the custom prompt template."""
    return PromptTemplate(CUSTOM_PROMPT_TEMPLATE)


def rephrase_query_function(query: str, index: VectorStoreIndex) -> str:
    """Rephrase the user query to be more specific and search-friendly."""
    rephrase_prompt = """Rephrase the following question to be more specific and searchable, 
    focusing on key taxation terms and concepts: {query}"""

    query_engine = index.as_query_engine()
    rephrased = query_engine.query(rephrase_prompt.format(query=query))

    return f"User query: {query}\nRephrased query: {rephrased}"


def retrieve_relevant_nodes(query: str, index: VectorStoreIndex) -> list:
    """Retrieve relevant nodes from the index based on the query."""
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(query)
    return nodes


if __name__ == "__main__":
    # Initialize the language model
    llm = initialize_llama_index_llm(OPENAI_API_KEY)
    client = initialize_llm(OPENAI_API_KEY)

    # Create the index
    index = initialize_index(QDRANT_URI, COLLECTION_NAME, llm)

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

    **Example Questions You Can Ask:**
    - "What are the GST requirements for small businesses?"
    - "How do I calculate capital gains tax on an investment property?"
    - "What superannuation contributions can I claim as a tax deduction?"

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

        rephrase_query: str = rephrase_query_function(query, index)

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
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                    + [{"role": "user", "content": formatted_prompt}],
                    stream=True,
                )

            response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})


# %%
