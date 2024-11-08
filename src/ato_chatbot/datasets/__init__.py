import json
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import pymongo


class Dataset(ABC):
    @abstractmethod
    def read_data(self, data_path: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        pass


class MongoDataset(Dataset):
    def __init__(self, mongod_config: dict, df: Optional[pd.DataFrame] = None) -> None:
        self.mongod_config = mongod_config
        self.df = df

    def read_data(self) -> pd.DataFrame:
        if self.df is None:
            # instanlize a client
            client = pymongo.MongoClient(self.mongod_config["MONGO_URI"])
            db = client[self.mongod_config["MONGO_DB"]]
            collection = db[self.mongod_config["MONGO_COLLECTION"]]

            # extract the data from mongodb, turn into a pandas dataframe
            results = collection.find()

            # turn into a pandas dataframe
            self.df = pd.DataFrame(results)

    def write_data(self) -> None:
        if self.df is not None:
            # instanlize a client
            client = pymongo.MongoClient(self.mongod_config["MONGO_URI"])
            db = client[self.mongod_config["MONGO_DB"]]
            collection = db[self.mongod_config["MONGO_COLLECTION"]]

            # remove the collection if it exists
            if self.mongod_config["MONGO_COLLECTION"] in db.list_collection_names():
                db.drop_collection(self.mongod_config["MONGO_COLLECTION"])

            # write the data to mongodb
            operations = [
                pymongo.UpdateOne(
                    {"og:url": row["og:url"]},
                    {"$setOnInsert": row.to_dict()},
                    upsert=True,
                )
                for _, row in self.df.iterrows()
                if row["og:url"] is not None
            ]
            collection.bulk_write(operations)
        else:
            raise ValueError("No data to write")

    def to_pandas(self) -> pd.DataFrame:
        return self.df


class LocalDataset(Dataset):
    def __init__(self, data_path, df: Optional[pd.DataFrame] = None) -> None:
        self.data_path = data_path
        self.df = df

    def read_data(self) -> pd.DataFrame:
        if self.df is None:
            with open(self.data_path, "r") as f:
                data = json.load(f)
                data = [
                    dict(markdown=item["markdown"], **item["metadata"]) for item in data
                ]

                # turn into a pandas dataframe
                self.df = pd.DataFrame(data)

        return self.df

    def to_pandas(self) -> pd.DataFrame:
        return self.df
