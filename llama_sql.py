from llama_index.llms.anthropic import Anthropic
from llama_index.core import Settings
from dotenv import load_dotenv
from llama_index.core import SQLDatabase, SimpleDirectoryReader, Document
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.indices.struct_store import SQLTableRetrieverQueryEngine
from sqlalchemy import create_engine, MetaData
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

load_dotenv()

tokenizer = Anthropic().tokenizer
Settings.tokenizer = tokenizer


llm = Anthropic(model="claude-3-haiku-20240307")
Settings.llm = llm
engine = create_engine("postgresql://postgres:postgres@localhost:5432/deez")
metadata_obj = MetaData()

sql_db = SQLDatabase(engine, include_tables=["jobdata"])
query_engine = NLSQLTableQueryEngine(sql_db)
response = query_engine.query("Which job descriptions contain AWS? give 5 IDs")


print(response)
print(response.metadata)
print(response.metadata['result'])


response_2 = query_engine.query(f"What are the job titles for these ids: {str(response.metadata['result'])}")
print(response_2)
print(response_2.metadata)
print(response_2.metadata['result'])
