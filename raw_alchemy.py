from sqlalchemy import MetaData, create_engine, inspect
from sqlalchemy.sql import text
from dotenv import load_dotenv
import os
from langchain_community.chat_models.litellm import ChatLiteLLM
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate

load_dotenv()


class Entity(BaseModel):
    entity_name: str
    relationships: List[str]
    validation_steps: List[str] = Field(default=None)


class Tables(BaseModel):
    tables: List[str]
    entities: List[Entity]
    solution_steps: List[str]


conn = create_engine(os.getenv("POSTGRES_CONNECTION"))
meta = inspect(conn)
tables = meta.get_table_names()


def get_columns(table_names):
    table_cols = {}
    for table in table_names:
        table_cols[table] = meta.get_columns(table)
    return table_cols


def build_prompt(table_cols):
    prompt = f""
    for k in table_cols:
        prompt += f"==========\n<table_name>{k}</table_name>\n\n<table_columns>{table_cols[k]}</table_columns>\n\n"
        prompt += f"============\n\n"
    return prompt


user_query = "Return the 10 most recent observations like claude 3 haiku. I want the input from these."
table_output_parser = PydanticOutputParser(pydantic_object=Tables)
query_analysis = """
Identify and list key entities in the query.
For each entity, validate and gather additional info from the database. Document findings.
Analyze relationships and dependencies between entities. Describe them.
Outline solution steps based on the information and analysis.
List any assumptions made and constraints to consider.
Define expected output format and any additional requirements.
Determine next steps for implementation and validation, and any further info needed.
"""
# 2 Step process
choose_table_prompt = "{query_analysis}\n\n<table_names>{tables}</table_names>\n\n<query>{query}</query>\n\n<format_instructions>{format_instructions}</format_instructions> Only output json and nothing else"
tables_prompt_template = PromptTemplate(
    template=choose_table_prompt,
    input_variables=["tables", "query"],
    partial_variables={
        "format_instructions": table_output_parser.get_format_instructions(),
        "query_analysis": query_analysis,
    },
)

llm = ChatLiteLLM(
    model="bedrock/anthropic.claude-3-haiku-20240307-v1:0", max_tokens=4000
)

chain = tables_prompt_template | llm | table_output_parser
output = chain.invoke({"tables": str(tables), "query": user_query})

# Get the columns from the returned list
selected_table_cols = get_columns(output.tables)
table_cols_prompt = build_prompt(selected_table_cols)

output_parser = StrOutputParser()
base_prompt = "Generate a valid postgres SQL query based on the given table information and query. Only output the valid postgres sql query and nothing else."
prompt_template = PromptTemplate(
    template="{instruction}\n\n{sql_prompt}\n\n<query>{query}</query>",
    input_variables=["instruction", "sql_prompt", "query"],
)

query_chain = prompt_template | llm | output_parser
out_2 = query_chain.invoke(
    {
        "instruction": base_prompt,
        "sql_prompt": str(table_cols_prompt),
        "query": user_query,
    }
)

with conn.connect() as c:
    statement = text(out_2)
    rs = c.execute(statement)
    for row in rs:
        print(row)
