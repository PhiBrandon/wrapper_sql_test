import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_anthropic import ChatAnthropic
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
import inspect

load_dotenv()


class SQLResponse(BaseModel):
    question: str
    sql_query: str


def main():
    # # postgresql://userame:password@localhost:5432/database_name
    db = SQLDatabase.from_uri("# postgresql://userame:password@localhost:5432/database_name")

    # Initialize the model
    model = ChatAnthropic(model="claude-3-haiku-20240307")

    db.get_table_info(["jobdata"])
    prompt = """
    You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question.\\nUnless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.\\nNever query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.\\nPay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.\\nPay attention to use CURRENT_DATE function to get the current date, if the question involves "today".Pay attention to generate queries that would be the most efficient based on the user question and the columns provided.\\n\\nHere's a JSON schema to follow: {pydantic_format} Output a valid JSON object but do not repeat the schema. . Only use the following tables:\\n{table_info}\\n\\nQuestion: {input}. Only output the valid JSON response. Do not include any other text or explanation in your response. If the questions contains arrays, make sure the final output that contains it does not have them.
    """

    custom_sql_prompt = PromptTemplate(
        input_variables=["input", "table_info", "pydantic_format", "top_k"],
        template=prompt,
    )

    chain = (
        create_sql_query_chain(model, db, prompt=custom_sql_prompt) | JsonOutputParser()
    )
    questions = [
        "What is the distribution of job positions across different locations?",
        "Which locations have the highest number of job postings?",
        "How many unique job positions are there in the table?",
        "What is the most common job position in the table?",
        "What is the average length of the job descriptions?",
        "Are there any job descriptions that contain specific keywords, such as 'remote', 'data', or 'healthcare'?",
        #"How many job postings were made today, and how does that compare to previous days?",
        #"What is the oldest job posting in the table, and how long ago was it posted?",
        "Is there a correlation between the length of the job description and the location of the job?",
        "Are there any job positions that are more likely to be remote compared to others?",
        #"What is the distribution of job postings across different days of the week?",
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
        response = chain.invoke(
            {
                "question": question,
                "pydantic_format": SQLResponse.model_json_schema(),
                "top_k": 5,
            }
        )
        print(response)

        valid_query = SQLResponse.model_validate(response)
        # print(valid_query.sql_query)
        query_output_1 = db.run(valid_query.sql_query)
        print(query_output_1)
    for q in questions:
        get_response(q)



    response_followup = chain.invoke(
        {
            "question": f"What are the job titles for these descriptions: {str(query_output_1)}",
            "pydantic_format": SQLResponse.model_json_schema(),
            "top_k": 5,
        }
    )
    print(response_followup)

    valid_query_2 = SQLResponse.model_validate(response_followup)
    print(valid_query_2.sql_query)

    query_output_2 = db.run(valid_query_2.sql_query)
    print(query_output_2)


if __name__ == "__main__":
    main()
