from anthropic import Anthropic
from sqlalchemy import create_engine, MetaData
from dotenv import load_dotenv

load_dotenv()

db = create_engine("postgresql://postgres:postgres@localhost:5432/deez")

conn = db.connect()

metadata_obj = MetaData()
info = metadata_obj.reflect(db)
tbl = metadata_obj.tables['jobdata']
tbl.columns.keys()