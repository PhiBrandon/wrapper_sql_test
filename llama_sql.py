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

    # postgresql://userame:password@localhost:5432/database_name
    engine = create_engine("postgresql://userame:password@localhost:5432/database_name")
    metadata_obj = MetaData()

    sql_db = SQLDatabase(engine, include_tables=["jobdata"])
    query_engine = NLSQLTableQueryEngine(sql_db)

    questions = [
        "What is the distribution of job positions across different locations?",
        "Which locations have the highest number of job postings?",
        "How many unique job positions are there in the table?",
        "What is the most common job position in the table?",
        "What is the average length of the job descriptions?",
        "Are there any job descriptions that contain specific keywords, such as 'remote', 'data', or 'healthcare'?",
        "How many job postings were made today, and how does that compare to previous days?",
        "What is the oldest job posting in the table, and how long ago was it posted?",
        "Is there a correlation between the length of the job description and the location of the job?",
        "Are there any job positions that are more likely to be remote compared to others?",
        "What is the distribution of job postings across different days of the week?",
        "Are there any patterns in the job IDs, such as specific prefixes or suffixes?",
        "What is the average number of words in the job position names?",
        "Are there any job positions that have multiple openings listed in the table?",
        "How many job postings mention specific programming languages or technologies in their descriptions?",
        "Is there a correlation between the job position and the likelihood of it being a remote position?",
        "Are there any job descriptions that mention specific benefits or perks?",
        "What is the distribution of job postings across different industries or sectors?",
        "Are there any job positions that have similar descriptions, potentially indicating duplicate postings?",
        "How many job postings have missing or incomplete information in any of the columns?",
    ]

    def get_response(question: str):
        # Original test - "Which job descriptions contain AWS? give 5 IDs"
        response = query_engine.query(question)
        print(response.metadata)
        print(response)

    for q in questions:
        get_response(q)

    follow_up = query_engine.query(
        "Where are the openings for Azure Data Engineer located?"
    )
    print(follow_up)
    """ response_2 = query_engine.query(
        f"What are the job titles for these ids: {str(response.metadata['result'])}"
    )
    print(response_2)
    print(response_2.metadata)
    print(response_2.metadata["result"]) """


if __name__ == "__main__":
    main()
