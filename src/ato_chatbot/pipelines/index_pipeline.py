# %%
# libs
import os
from typing import Union

import numpy as np
import pandas as pd
## llama index stuff
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from typing_extensions import Annotated
## zenml stuff
from zenml import Model, get_step_context, log_step_metadata, pipeline, step
from zenml.logger import get_logger

## internal
from ato_chatbot.datasets import LocalDataset, MongoDataset

## change pandas display options
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)


## logger
logger = get_logger(__name__)


# %%
# PARAMETERS

## mongodb parameters, reading it from env variables

MONGO_RAW_CONFIG = {
    "MONGO_URI": os.getenv("MONGO_URI"),
    "MONGO_DB": os.getenv("MONGO_DB_RAW"),
    "MONGO_COLLECTION": "v1.1",
}

MONGO_CLEANED_CONFIG = {
    "MONGO_URI": os.getenv("MONGO_URI"),
    "MONGO_DB": os.getenv("MONGO_DB"),
    "MONGO_COLLECTION": os.getenv("MONGO_COLLECTION"),
}

## find the repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

## local data file, at the root of the repo
DATA_PATH = os.path.join(REPO_ROOT, "data/documents.json")

## zenml setup
MODEL_NAME = "ato-chatbot-index"
MODEL_VERSION = None

## INDEX parameters
NUM_DOCS = -1
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
QDRANT_URI = "http://127.0.0.1:6333"
NUM_WORKERS = 10

# %%


# first part is data ingesting and cleaning
@step(enable_cache=False)
def load_data(data_path: str) -> Annotated[LocalDataset, "raw_dataset"]:
    local_dataset = LocalDataset(data_path)
    local_dataset.read_data()

    return local_dataset


@step(enable_cache=False)
def load_mongo_data(mongod_config: dict) -> Annotated[MongoDataset, "raw_dataset"]:
    mongo_dataset = MongoDataset(mongod_config)
    mongo_dataset.read_data()
    return mongo_dataset


@step(enable_cache=False)
def clean_and_upload_data(
    dataset: Union[LocalDataset, MongoDataset], mongod_config: dict
) -> Annotated[MongoDataset, "cleaned_dataset"]:
    cols_keep = [
        "markdown",
        "og:url",
        "title",
        "description",
        "og:type",
        "keywords",
        "language",
    ]

    df = dataset.to_pandas()

    df = df[cols_keep]

    # remove the url contain "other-language"
    df2 = df[df["og:url"].str.contains("other-language") == False]

    # First remove trailing slashes to standardize the URLs
    df2.loc[:, "og:url"] = df2["og:url"].str.rstrip("/")

    # Then filter for URLs with exactly 2 subpages
    df2 = df2[df2["og:url"].str.count("/") >= 4]

    # themes
    df2.loc[:, "url_clenaed"] = df2["og:url"].str.replace("https://www.ato.gov.au/", "")
    df2.loc[:, "themes"] = df2["url_clenaed"].str.split("/")

    # cleaning the data
    df2 = df2[df2["themes"].apply(len) <= 7]

    # max level of the themes
    max_lvl = df2["themes"].apply(len).max()

    # extract the sub pages
    for lvl in range(0, max_lvl):
        df2.loc[:, f"sub_page_{lvl}"] = df2.loc[:, "themes"].apply(
            lambda x: x[lvl] if len(x) > lvl else None
        )

    # drop all nan columns
    df2.dropna(axis=1, how="all", inplace=True)

    # clean keywords
    df2.loc[:, "keywords"] = (
        df2["keywords"]
        .str.split(",")
        .map(lambda x: [s.strip() for s in x] if isinstance(x, list) else np.nan)
    )

    # write to mongodb
    mongo_dataset = MongoDataset(mongod_config, df2)
    mongo_dataset.write_data()

    return mongo_dataset


# %%

# building index


# todo
@step
def build_index(
    dataset: MongoDataset,
    qdrant_uri: str,
    chunk_size: int,
    chunk_overlap: int,
    num_docs: int,
    using_qdrant: bool = True,
    num_workers: int = 4,
) -> None:
    """build the index and store the index Qdrant, expose chunk_size, chunk_overlap, and num_docs"""

    # access the model name from the pipeline step
    model_name = get_step_context().model.name

    # version of model
    model_version = get_step_context().model.version

    # using model name and version to create a collection name
    collection_name = f"{model_name}-{model_version}"

    # client
    client = QdrantClient(url=qdrant_uri)

    # create two sets of vectors fro hybrid search

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "text-dense": models.VectorParams(
                size=1536,  # openai vector size
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "text-sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )

    vector_store = QdrantVectorStore(
        client=client, collection_name=collection_name, enable_hybrid=True
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # create the pipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            TitleExtractor(),
            OpenAIEmbedding(),
        ]
    )

    # reading data from mongo
    dataset.read_data()
    df = dataset.to_pandas()

    # using subset for poc
    df = df.head(num_docs) if num_docs > 0 else df

    # creating documents
    documents = [
        Document(
            text=row["markdown"],
            metadata={"source": row["og:url"], "title": row["title"]},
        )
        for _, row in df.iterrows()
    ]

    # build index from nodes
    nodes = pipeline.run(documents=documents, num_workers=num_workers)

    # add nodes to vector store
    if using_qdrant:
        ato_vector_index = VectorStoreIndex(
            nodes=nodes, storage_context=storage_context
        )
    else:
        ato_vector_index = VectorStoreIndex(nodes=nodes)

    # metadata
    log_step_metadata(
        metadata={
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_docs": df.shape[0],
            "qdrant_collection_name": collection_name,
            "using_qdrant": using_qdrant,
        },
    )


# %%

# define model

model = Model(
    name=MODEL_NAME,
    version=MODEL_VERSION or None,
    description="A index model for the ATO chatbot",
    tags=["ato", "chatbot", "index"],
)


@pipeline
def chatbot_index_pipeline(
    data_path: str,
    qdrant_uri: str,
    chunk_size: int,
    chunk_overlap: int,
    num_docs: int,
    using_qdrant: bool,
    num_workers: int,
    use_json: bool = True,
):
    if use_json:
        dataset = load_data(DATA_PATH)
    else:
        dataset = load_mongo_data(MONGO_RAW_CONFIG)

    mongo_dataset = clean_and_upload_data(
        dataset,
        MONGO_CLEANED_CONFIG,
    )

    build_index(
        mongo_dataset,
        qdrant_uri,
        chunk_size,
        chunk_overlap,
        num_docs,
        using_qdrant,
        num_workers,
    )


# %%


if __name__ == "__main__":
    # Run the pipeline and configure some parameters at runtime
    pipeline_run = chatbot_index_pipeline.with_options(
        model=model,
    )(
        data_path=DATA_PATH,
        qdrant_uri=QDRANT_URI,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        num_docs=NUM_DOCS,
        using_qdrant=True,
        num_workers=NUM_WORKERS,
        use_json=False,
    )
