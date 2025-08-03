from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing import Literal, List

import asyncio
import pandas as pd
import os


# Load environment variables and specify model
_ = load_dotenv(find_dotenv())
groq_model = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0
MAX_INPUT_TEXT_LENGTH = 5_000    # words

llm = ChatGroq(model=groq_model, temperature=LLM_TEMPERATURE)

# Load the data and convert it to langchain documents
file = os.getenv("SOURCE_FILE")
df = pd.read_parquet(file)
source_file = file.split(os.sep)[-1]

langchain_documents = []
for index, content in df["text"].items():
    doc = Document(
        page_content=content,
        metadata={"source": source_file, "index": index, "num_words": len(content.split())}
    )
    langchain_documents.append(doc)

#print(sum(doc.metadata["num_words"] for doc in langchain_documents))

# Initial summary
initial_summary_template = """
You're a smart writing assistant. Your responses are always concise and to the point
Summarize the following text(s).

Text(s): {context}
"""
initial_summary_prompt = ChatPromptTemplate.from_template(initial_summary_template)
initial_summary_chain = initial_summary_prompt | llm | StrOutputParser()

num_docs = 5  # How many documents to summarize
#initial_summary = initial_summary_chain.invoke({"context": langchain_documents[:num_docs]})
#print(f"Original text: {langchain_documents[0].page_content}\n\n")
#print(f"Summarized text: {initial_summary}\n\n")
#print(f"Original texts length: {sum(langchain_documents[i].metadata.get('num_words') for i in range(num_docs))} words.")
#print(f"Summarized text length: {len(initial_summary.split())} words.")

# Update the summary from more documents
summary_update_template = """
You're a smart and helpful writing assistant.
Given an existing summary, and some additional text, you first summarize the additional text, 
and then attach create an overall final summary of both the original and new summaries.
Ensure that the important elements of each constituent summary are captured in the final summary.

Existing summary up to this point:
{existing_summary}

New text:
------------
{additional_text}
------------

Remember to keep your response with concise.
Your response should only contain the final summary.
"""

summary_update_prompt = ChatPromptTemplate.from_template(summary_update_template)
summary_update_chain = summary_update_prompt | llm | StrOutputParser()

# Define pydantic data model to hold the state of the graph
# Including the documents, the summary, and an index to track the current position in the dataset
class State(BaseModel):
    documents: List[Document] = Field(description="List of the documents to summarize.")
    index: int = Field(description="Index of the current document.", default=0)
    summary: str = Field(description="The current summary of the documents.", default="")

# Define node functions
async def generate_initial_summary(state: State, config: RunnableConfig):
    initial_summary = await initial_summary_chain.ainvoke(
        {"context": state.documents[0]},
        config
    )
    return {"summary": initial_summary, "index": 1}

async def update_summary(state: State, config: RunnableConfig):
    additional_document = state.documents[state.index]
    updated_summary = await summary_update_chain.ainvoke(
        {"existing_summary": state.summary, "additional_text": additional_document},
        config
    )
    return {"summary": updated_summary, "index": state.index + 1}

# Logic to exit application or further update the summary
def should_update_summary(state: State) -> Literal["update_summary", END]:
    if state.index >= len(state.documents):
        return END
    else:
        return "update_summary"

# Define the graph
summary_update_graph = StateGraph(State)
summary_update_graph.add_node("generate_initial_summary", generate_initial_summary)
summary_update_graph.add_node("update_summary", update_summary)

summary_update_graph.add_edge(START, "generate_initial_summary")
summary_update_graph.add_conditional_edges("generate_initial_summary", should_update_summary)
summary_update_graph.add_conditional_edges("update_summary", should_update_summary)

app = summary_update_graph.compile()

# Run the graph
async def run_app(state: State):
    async for step in app.astream(
            state,
            stream_mode="values"
    ):
        if summary := step.get("summary"):
            print(f"Current summary:\n---------------\n {summary}")

asyncio.run(run_app(State(documents=langchain_documents[:num_docs])))
