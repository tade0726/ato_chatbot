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
You are an expert assistant specialising in Australian Taxation Office (ATO) matters. Your primary role is to provide accurate, factual information about Australian taxation by carefully analysing and synthesising the provided context. You must NOT provide financial advice or personalized recommendations.

Instructions for Response:
1. Focus ONLY on Australian taxation and ATO-related matters
2. Analyse all provided context thoroughly before answering
3. Only use information explicitly present in the context
4. Cite specific sections or sources when making statements
5. If information is partial or unclear, acknowledge the limitations
6. For any questions seeking financial advice or recommendations, respond with a clear disclaimer and redirect to appropriate professionals
7. Never provide personalized recommendations or "should" statements
8. Use Australian English in all responses

Acceptable Topics (Factual Information Only):
- Australian tax laws and regulations
- ATO services and requirements
- Tax deductions and returns
- GST and income tax matters
- Business tax obligations in Australia
- Taxation aspects of superannuation
- Tax-related financial reporting requirements

Explicitly Reject and Redirect (with appropriate disclaimer):
- Requests for financial advice or recommendations
- Questions seeking personalized investment strategies
- Queries about what investments to make
- Requests for opinions on financial products
- Questions about future market performance
- Requests for personalized retirement planning
- Questions about which superannuation fund to choose
- Personal financial decisions or "should I" questions

---
**Response Structure:**

[Provide a clear, context-based answer that directly addresses the user's tax-related question]
Each statement should include a source reference link in parentheses.

**Key Information:**
- **Verified Facts:** 
  • [Fact 1] (Source: [URL])
  • [Fact 2] (Source: [URL])
  • [Additional facts with source links]

- **Requirements:** 
  • [Requirement 1] (Source: [URL])
  • [Requirement 2] (Source: [URL])
  • [Additional requirements with source links]

- **Important Dates/Deadlines:** 
  • [Date/Deadline 1] (Source: [URL])
  • [Date/Deadline 2] (Source: [URL])
  • [Additional dates with source links]

**Source References:**
Each source used must be listed with:
- Source URL: [URL]
- Information used: [Brief description]
- Relevance to query: [Brief explanation]

---
**Confidence Level:**
- High: All tax-related information directly supported by context
- Partial: Some aspects need additional verification
- Limited: Insufficient context to fully answer

*Note: This response is based solely on official ATO documentation. For personalized tax advice, please consult a registered tax agent. For financial advice, please consult a licensed financial adviser.*
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
        Analyze the query and determine if it requests factual Australian taxation/ATO information. Return a JSON response with a boolean decision and reasoning.

        APPROVE (return true) if query seeks factual information about:
        1. Australian Tax Laws and Procedures
           - Tax legislation and regulations
           - ATO administrative processes
           - Lodgment requirements
           - Tax rates and thresholds
        
        2. Tax Documentation and Reporting
           - Required forms and documentation
           - Record keeping requirements
           - Reporting deadlines
           - Tax file number (TFN) matters
        
        3. Specific Tax Categories
           - Income tax calculations
           - GST requirements
           - Capital gains tax facts
           - Fringe benefits tax
           - Business tax obligations
           - Superannuation tax treatment
        
        REJECT (return false) if query involves:
        1. Financial Advice
           - Investment recommendations
           - Financial planning suggestions
           - Portfolio management
           - Risk assessment advice
        
        2. Out of Scope
           - Non-Australian tax matters
           - General financial topics
           - Personal opinions
           - Off-topic queries
           - Market predictions
        
        Response Format:
        {
            "intention": boolean,
            "reason": "Specific reason for classification based on above criteria"
        }
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

        # clean the response text, if ```json``` is present, remove it
        if response_text.startswith("```json"):
            response_text = response_text.lstrip("```json").rstrip("```")

        # Parse the response text as JSON
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
    - **Consultation:** For personalised tax advice, please reach out to a certified tax professional 👨‍💼👩‍💼
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

    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Your message"):
        logger.debug(f"Query: {query}")

        # Increment interaction counter and show feedback reminder
        st.session_state.interaction_count += 1

        # Check query intention
        if not (intention := intetion_recognition_function(client, query)):
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                message = """I apologize, but I can only provide factual information about Australian taxation and ATO matters. I cannot assist with regulated financial services or personal recommendations.

I can help with:
- Australian tax laws and ATO procedures
- Tax returns, deductions, and obligations
- Tax-related aspects of superannuation
- Factual tax information for investments and assets

I cannot provide assistance with:
- Financial product advice
- Investment recommendations or strategies
- Market-making or securities trading
- Fund management or investment schemes
- Superannuation fund selection
- Personal financial planning
- "Should I" questions about financial decisions
- Non-Australian tax matters
- General chat or off-topic queries

For regulated financial services, please consult:
- A licensed Financial Adviser (find one through the [Financial Advisers Register](https://moneysmart.gov.au/financial-advice/financial-advisers-register))
- A registered Tax Agent (search the [Tax Practitioners Board Register](https://www.tpb.gov.au/registrations_search))

How can I assist you with factual tax-related information?"""
                st.warning(message)
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
                    response = client.chat.completions.create(
                        model=st.session_state["openai_model"],
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
                        + [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ]
                        + [{"role": "user", "content": formatted_prompt}],
                    )
                    st.markdown(response.choices[0].message.content)

            st.session_state.messages.append(
                {"role": "assistant", "content": response.choices[0].message.content}
            )

        if (
            st.session_state.interaction_count % 2 == 0
        ):  # Show reminder every 2 interactions
            st.info(
                "📝 Enjoying the chat? Please take a moment to share your feedback! "
                "[Fill out our quick survey](https://forms.gle/kS5qDPY1dHWAA9MQ9)\n\n"
                "💡 Leave your email in the survey to stay updated on future developments!"
            )


# %%
