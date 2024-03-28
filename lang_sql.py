import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_anthropic import ChatAnthropic
from langchain.chains import create_sql_query_chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel

load_dotenv()


class SQLResponse(BaseModel):
    question: str
    sql_query: str


def main():
    db = SQLDatabase.from_uri("postgresql://username:password@localhost:5432/database_name")

    # Initialize the model
    model = ChatAnthropic(model="claude-3-haiku-20240307")

    prompt = """
    You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question.\\nUnless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.\\nNever query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.\\nPay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.\\nPay attention to use CURRENT_DATE function to get the current date, if the question involves "today".Pay attention to generate queries that would be the most efficient based on the user question and the columns provided.\\n\\nHere's a JSON schema to follow: {pydantic_format} Output a valid JSON object but do not repeat the schema. . Only use the following tables:\\n{table_info}\\n\\nQuestion: {input}. Only output the valid JSON response. Do not include any other text or explanation in your response. If the questions contains arrays, make sure the final output that contains it does not have them.
    """

    custom_sql_prompt = PromptTemplate(
        input_variables=["input", "table_info", "pydantic_format", "top_k"],
        template=prompt,
    )

    chain = create_sql_query_chain(model, db, prompt=custom_sql_prompt) | JsonOutputParser()

    response = chain.invoke(
        {
            "question": "Which job descriptions contain AWS? give the IDs",
            "pydantic_format": SQLResponse.model_json_schema(),
            "top_k": 5,
        }
    )
    print(response)

    valid_query = SQLResponse.model_validate(response)
    print(valid_query.sql_query)

    query_output_1 = db.run(valid_query.sql_query)
    print(query_output_1)

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