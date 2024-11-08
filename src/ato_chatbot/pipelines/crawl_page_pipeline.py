# %% libs


# logging
import logging
# env
import os
# time
import time

# pandas
import pandas as pd
# fire
from firecrawl import FirecrawlApp
from typing_extensions import Annotated
## zenml stuff
from zenml import Model, get_step_context, log_step_metadata, pipeline, step
from zenml.logger import get_logger

# datasets
from ato_chatbot.datasets import MongoDataset

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

# requests
import requests

# %% parameters

TARGET_URL = "https://www.ato.gov.au"
NUM_PAGES = 1991

## model parameters
MODEL_NAME = "crawl-page-pipeline"
MODEL_VERSION = "v1.1"

## mongodb parameters, reading it from env variables
MONGOD_CONFIG = {
    "MONGO_URI": os.getenv("MONGO_URI"),
    "MONGO_DB": os.getenv("MONGO_DB_RAW"),
    "MONGO_COLLECTION": MODEL_VERSION,
}

CRAWL_ID = {
    "id": "d7051c2d-b049-41f8-9f8a-3142a5559441",
}

# %% steps


@step
def submit_crawl_request(target_url: str, num_pages: int, crawl_id: str = None) -> dict:
    if crawl_id is not None:
        return {
            "id": crawl_id,
            "status": "completed",
        }

    # init the crawl
    app = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])

    # Crawl a website:
    crawl_status = app.async_crawl_url(
        target_url,
        params={
            "limit": 1000 if num_pages == -1 else num_pages,
            "scrapeOptions": {"formats": ["markdown", "html"]},
        },
    )

    return crawl_status


@step(enable_cache=False)
def save_crawl_results(
    crawl_status: dict, mongod_config: dict
) -> Annotated[MongoDataset, "scraped_pages"]:
    # checking the status of the crawl
    crawl_id = crawl_status["id"]

    # init the crawl
    app = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])

    while True:
        resp = app.check_crawl_status(crawl_id)
        if resp["status"] == "completed":
            break
        time.sleep(30)

    # unpack the data
    data = resp["data"]

    if resp.get("next") is not None:
        full_data = requrest_full_data(resp["next"])
        data.extend(full_data)

    logger.debug(f"Found {len(data)} pages in the crawl")

    if not data:
        logger.error("No data found in the crawl")
        raise ValueError("No data found in the crawl")

    data2 = [
        dict(markdown=item["markdown"], **item["metadata"])
        for item in data
        if item is not None
    ]

    # turn into a dataframe
    df = pd.DataFrame(data2)
    dataset = MongoDataset(mongod_config, df)
    dataset.write_data()

    return dataset


def requrest_full_data(url: str):
    try:
        headers = {"Authorization": f"Bearer {os.environ['FIRECRAWL_API_KEY']}"}
        response = requests.request("GET", url, headers=headers)

        data = []
        batch_id = 0

        while True:
            batch = response.json().get("data", [])
            data.extend(batch)

            if response.json().get("next") is None:
                break

            response = requests.request("GET", response.json()["next"], headers=headers)

            logger.debug(f"Batch {batch_id} completed")
            batch_id += 1

        return data

    except Exception as e:
        logger.error(f"Error requesting full data: {e}")
        logger.error(f"Response: {response.json()}")
        raise e


# %% pipeline

model = Model(
    name=MODEL_NAME,
    version=MODEL_VERSION or None,
    description="Extracts pages from the ATO website",
    tags=["ato", "crawl", "data"],
)


@pipeline
def crawl_page_pipeline(
    mongod_config: dict,
    target_url: str,
    num_pages: int,
    crawl_id: dict,
):
    # get the crawl id
    crawl_id = crawl_id.get("id", None) if crawl_id else None

    # submit crawl request
    crawl_status = submit_crawl_request(target_url, num_pages, crawl_id)

    # save crawl results
    _ = save_crawl_results(crawl_status, mongod_config)


# %%

if __name__ == "__main__":
    dataset = crawl_page_pipeline.with_options(model=model)(
        MONGOD_CONFIG, TARGET_URL, NUM_PAGES, CRAWL_ID
    )
