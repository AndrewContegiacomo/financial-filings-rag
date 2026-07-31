"""
Tool-calling agent over the filings corpus.

The model decides which tool to call and with what parameters; the tools
do retrieval and arithmetic. Plain RAG remains the right path for
narrative questions ("what risks does X face") — this adds the ability
to answer quantitative and comparative ones without the model doing math.
"""
import json

from rag.llm_client import call_llm
from rag.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

MAX_TURNS = 5  # guard against a model looping on tool calls

SYSTEM = """You are a financial analyst assistant working with SEC
filings for Apple (AAPL), Microsoft (MSFT), JPMorgan Chase (JPM) and
Pfizer (PFE).

Use the tools to obtain figures. Never calculate a value yourself: if a
question involves a change, growth, or difference, call compare_periods
and report what it returns.

If a tool rejects a lookup as implausible, do not work around it by
calling lookup_metric for the same metric — retry with a more specific
metric name, or report that the figure could not be established.

Answer only from tool results. Cite the source tag each figure came
from. If a tool reports a figure was not found, say so plainly."""


def run(question: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_TURNS):
        msg = call_llm(messages, tools=TOOL_SCHEMAS, tool_choice="auto")
        if msg is None:
            return ("The language model is temporarily unavailable. "
                    "Please try again in a moment.")

        messages.append(msg)

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            if verbose:
                print(f"  → {call.function.name}({args})")
            result = TOOL_FUNCTIONS[call.function.name](**args)
            if verbose:
                print(f"    {json.dumps(result)[:200]}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return "Could not complete the request within the tool-call limit."


if __name__ == "__main__":
    print("Agent ready. Empty line to quit.\n")
    while True:
        q = input("Question> ").strip()
        if not q:
            break
        print()
        print(run(q))
        print()