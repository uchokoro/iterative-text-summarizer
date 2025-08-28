"""
Prompt templates for the incremental summarization agent.

This module exposes two LangChain `ChatPromptTemplate`s:

- `initial_summary_prompt`
    A concise-first instruction to summarize an input passage.
    Expects the variable: `context` (str) — the text to summarize.

- `summary_update_prompt`
    A two-step instruction that (1) summarizes *additional* text and then
    (2) produces a single, concise, combined summary that integrates the
    prior summary with the new material.
    Expects the variables:
      - `existing_summary` (str) — the running summary so far
      - `additional_text` (str) — new text to fold into the summary

Dependencies
- langchain

Usage (example)
    from langchain_groq import ChatGroq
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)

    # Initial summary
    chain_init = initial_summary_prompt | llm | StrOutputParser()
    s0 = chain_init.invoke({"context": "Some long text..."})

    # Update summary with more text
    chain_update = summary_update_prompt | llm | StrOutputParser()
    s1 = chain_update.invoke({
        "existing_summary": s0,
        "additional_text": "New material to include..."
    })

Notes
- For best results, pass raw text strings for `context` and `additional_text`
  (e.g., `doc.page_content`) rather than full `Document` objects.
- Prompts are written to encourage brevity; downstream chains should not add
  extra formatting beyond the final summary text.
"""

from langchain_core.prompts import ChatPromptTemplate

# Initial summary
initial_summary_template = """
You're a smart writing assistant. Your responses are always concise and to the point
Summarize the following text(s).

Text(s): {context}
"""

initial_summary_prompt = ChatPromptTemplate.from_template(initial_summary_template)

# Summary update
summary_update_template = """
You're a smart and helpful writing assistant.
Given an existing summary, and some additional text, you first summarize the additional text, 
and then create an overall final summary of both the original and new summaries.
Ensure that the important elements of each constituent summary are captured in the final summary.

Existing summary up to this point:
{existing_summary}

Additional text to summarize:
------------
{additional_text}
------------

Remember to keep your response concise.
Your response should only contain the final summary.
"""

summary_update_prompt = ChatPromptTemplate.from_template(summary_update_template)
