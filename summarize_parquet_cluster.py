"""
Incremental summarization agent for a Parquet-backed text corpus.

Overview
- Loads a Parquet file (path via SOURCE_FILE) whose "text" column holds input passages.
- Wraps each row as a LangChain `Document`, then uses a LangGraph workflow to build a
  running summary:
    1) `generate_initial_summary` creates the first summary from the first document.
    2) `update_summary` iteratively refines that summary with subsequent documents.
- A simple label-based router stops when an error occurs, `num_docs` is reached,
  or all documents are consumed. Progress prints to stdout after each update.

Local modules (required)
- data_models.py
  - Exposes `State` used by the graph. Expected fields include:
    - documents: list[Document]
    - index: int (defaults to 0)
    - summary: str (defaults to "")
    - is_error_encountered: bool (defaults to False)
    - error_description: str (defaults to "")
- prompts.py
  - Exposes `initial_summary_prompt` and `summary_update_prompt` used to drive the LLM.

Third-party dependencies
- pandas, pyarrow or fastparquet (for Parquet), python-dotenv
- langchain, langchain-groq (LLM client), langgraph

Environment
- SOURCE_FILE: path to a `.parquet` file containing a "text" column.
- GROQ_API_KEY: (required by `ChatGroq`) API key in your environment.

Key parameters
- `groq_model`, `LLM_TEMPERATURE`: LLM configuration for `ChatGroq`.
- `MAX_INPUT_TEXT_LENGTH`: per-document input cap before sending to the LLM.
- `num_docs`: upper bound on how many documents to summarize in this run.

Usage
    export SOURCE_FILE=/path/to/data.parquet
    export GROQ_API_KEY=your_key_here
    python script_name.py
"""

import asyncio
import pandas as pd
import os
import re
from copy import deepcopy
from data_models import State
from dotenv import find_dotenv, load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from prompts import initial_summary_prompt, summary_update_prompt
from typing import Literal


# Load environment variables and specify model
_ = load_dotenv(find_dotenv())
groq_model = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.0
MAX_INPUT_TEXT_LENGTH = 5_000    # words

llm = ChatGroq(model=groq_model, temperature=LLM_TEMPERATURE)

# Load the text data and convert it to langchain documents
file = os.getenv("SOURCE_FILE")
df = pd.read_parquet(file)
source_file = file.split(os.sep)[-1]

documents = []

for index, content in df["text"].items():
    doc = Document(
        page_content=content,
        metadata={"source": source_file, "index": index, "num_words": len(content.split())}
    )

    documents.append(doc)


# Chain for generating the summarizing a single document
initial_summary_chain = initial_summary_prompt | llm | StrOutputParser()

num_docs = 5  # Maximum number of documents to summarize

# Chain for updating the summary from more documents
summary_update_chain = summary_update_prompt | llm | StrOutputParser()

# Define agent node functions
def limit_document_text_length(document: Document, max_words: int = MAX_INPUT_TEXT_LENGTH) -> Document:
    r"""
    Return a NEW `Document` truncated to at most `max_words` words.
    The original `document` is NOT modified.

    - Words are detected as non-whitespace runs (`\S+`).
    - Preserves original formatting up to the cutoff.
    - Sets `metadata["num_words"]` on the returned copy to:
        * `max_words` if truncated
        * the actual word count otherwise
    """

    # Copy metadata so caller mutations don't affect the original
    meta = deepcopy(getattr(document, "metadata", {}))
    text = getattr(document, "page_content", "")

    if not isinstance(text, str):
        # Non-text content: just return a copy
        return Document(page_content=text, metadata=meta)

    total_words = 0
    cutoff_end = None

    for match in re.finditer(r"\S+", text):
        total_words += 1
        if total_words == max_words:
            cutoff_end = match.end()

    if total_words > max_words and cutoff_end is not None:
        truncated_text = text[:cutoff_end]
        meta["num_words"] = max_words
        return Document(page_content=truncated_text, metadata=meta)

    # No truncation needed; still return a fresh copy with accurate count
    meta["num_words"] = total_words
    return Document(page_content=text, metadata=meta)


async def generate_initial_summary(state: State) -> State:
    """Agent node to summarize the first document in a list of LangChain documents."""
    if state.documents is None or len(state.documents) == 0:
        state_update_dict = {
            "is_error_encountered": True,
            "error_description": "Error: Agent state contains no document to summarize!"
        }
    else:
        try:
            document = limit_document_text_length(state.documents[0])
            initial_summary = await initial_summary_chain.ainvoke(
                {"context": document}
            )
            state_update_dict = {"summary": initial_summary, "index": 1}
        except Exception as e:
            state_update_dict = {
                "is_error_encountered": True,
                "error_description": f"""
                Error encountered while generating the initial summary!\n{str(e)}
                """
            }

    return state.model_copy(update=state_update_dict)


async def update_summary(state: State) -> State:
    """Agent node to update the documents' summary based on the next document in the list."""
    if state.is_error_encountered:
        return state

    starting_index = state.index

    try:
        additional_document = limit_document_text_length(state.documents[starting_index])
        updated_summary = await summary_update_chain.ainvoke(
            {"existing_summary": state.summary, "additional_text": additional_document},
        )
        state_update_dict = {"summary": updated_summary, "index": state.index + 1}
    except Exception as e:
        state_update_dict = {
            "is_error_encountered": True,
            "error_description": f"""
            Error encountered while updating the summary with document {starting_index}!\n{str(e)}
            """
        }

    return state.model_copy(update=state_update_dict)

# Logic to exit application or further update the summary
def router(state: State) -> Literal["update_summary", "__end__"]:
    """
    Agentic workflow router.
    Evaluates whether to continue updating the summary based on additional documents or not.
    If an error had been encountered in a previous step of the agent's current run,
    or the given maximum number of documents, or all the documents, have already been summarized,
    it halts the summary update loop.
    """
    if state.is_error_encountered or state.index >= num_docs or state.index >= len(state.documents):
        return "__end__"

    return "update_summary"

# Define the agent-graph
summary_update_graph = StateGraph(State)
summary_update_graph.add_node("generate_initial_summary", generate_initial_summary)
summary_update_graph.add_node("update_summary", update_summary)

summary_update_graph.set_entry_point("generate_initial_summary")

summary_update_graph.add_conditional_edges(
    "generate_initial_summary",
    router,
    {"__end__": END, "update_summary": "update_summary"}
)
summary_update_graph.add_conditional_edges(
    "update_summary",
    router,
    {"__end__": END, "update_summary": "update_summary"}
)

summary_agent = summary_update_graph.compile()

# Run the graph
async def run_agent(state: State):
    async for step in summary_agent.astream(
            state,
            stream_mode="values"
    ):
        if summary := step.get("summary"):
            print(f"Current summary (next document index: {step.get("index")}):\n---------------\n {summary}")

asyncio.run(run_agent(State(documents=documents[:num_docs])))
