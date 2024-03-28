import os
from dotenv import load_dotenv
from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from llama_index.core import SQLDatabase, SimpleDirectoryReader, Document
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from sqlalchemy import create_engine, MetaData
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()


def main():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    tokenizer = Anthropic().tokenizer
    Settings.tokenizer = tokenizer

    llm = Anthropic(model="claude-3-haiku-20240307")
    Settings.llm = llm

    engine = create_engine("postgresql://username:password@localhost:5432/database_name")
    metadata_obj = MetaData()

    sql_db = SQLDatabase(engine, include_tables=["jobdata"])
    query_engine = NLSQLTableQueryEngine(sql_db)

    response = query_engine.query("Which job descriptions contain AWS? give 5 IDs")
    print(response)
    print(response.metadata)
    print(response.metadata['result'])

    response_2 = query_engine.query(
        f"What are the job titles for these ids: {str(response.metadata['result'])}"
    )
    print(response_2)
    print(response_2.metadata)
    print(response_2.metadata['result'])


if __name__ == "__main__":
    main()