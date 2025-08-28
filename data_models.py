"""
Data models for the summarization agent graph.

This module defines `State`, a Pydantic model used as the single message-passing
state in the LangGraph workflow. Nodes receive a `State`, produce a new one with
`state.model_copy(update=...)`, and the router decides whether to continue
iterating over the input documents or halt.

Dependencies
- langchain
- pydantic

Fields
- documents (list[Document]): The corpus to summarize (required).
- index (int): 0-based index of the NEXT document to process. Defaults to 0.
- summary (str): Running/cumulative summary BEFORE processing `index`. Defaults to "".
- is_error_encountered (bool): Sticky error flag; when True, the graph should halt. Defaults to False.
- error_description (str): Human-readable details when an error occurs. Defaults to "".

Notes
- Keep `index` in sync with how many documents have already been summarized.
- Treat `State` as immutable within nodes; return a copy via `model_copy(update=...)`.

Usage (example)
    from langchain_core.documents import Document
    docs = [Document(page_content="..."), Document(page_content="...")]
    state = State(documents=docs)  # index=0, empty summary, no error
"""

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class State(BaseModel):
    """Agent graph state data model"""
    documents: list[Document] = Field(
        description="List of the documents to summarize."
    )
    index: int = Field(
        description="Index of the next document to be summarized.",
        default=0
    )
    summary: str = Field(
        description="The cumulative summary of the documents before the most recently summarized one.",
        default=""
    )
    is_error_encountered: bool = Field(
        description="An indicator of whether an error occurred in a previous step of the agent's current run.",
        default=False
    )
    error_description: str = Field(
        description="Description of an encountered error in a previous step of the agent's current run.",
        default=""
    )
