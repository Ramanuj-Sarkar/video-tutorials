"""
demo_agent.py
A minimal LangChain agent with two tools: calculator and word_length.

Setup
-----
    pip install langchain langchain-classic langchain-anthropic

    # macOS / Linux
    export ANTHROPIC_API_KEY="sk-ant-..."

    # Windows (PowerShell)
    $env:ANTHROPIC_API_KEY = "sk-ant-..."

Run
---
    python demo_agent.py

Swap in a different model provider (e.g. OpenAI) by changing just the
"Model" section and ChatAnthropic import below — everything else (tools, prompt, agent,
executor) stays the same.
"""

from langchain_anthropic import ChatAnthropic
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool


# ---------------------------------------------------------------------
# 1) TOOLS — plain Python functions the model is allowed to call.
#    The docstring IS the tool description the model sees, so keep
#    it short and accurate.
# ---------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '23 * 17'."""
    try:
        # A restricted eval: no builtins, so this can't do anything
        # beyond arithmetic on the expression string it's given.
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as exc:
        return f"Error: {exc}"


@tool
def word_length(word: str) -> str:
    """Return the number of characters in a single word."""
    return str(len(word))


tools = [calculator, word_length]


# ---------------------------------------------------------------------
# 2) MODEL & PROMPT — the model, and the instructions that frame its job.
#    {agent_scratchpad} is where LangChain injects the tool calls and
#    tool results as the agent works through a question.
# ---------------------------------------------------------------------

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use tools when they "
                    "help you answer accurately."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


# ---------------------------------------------------------------------
# 3) AGENT — the model, wired up to decide which tool to call and when.
# ---------------------------------------------------------------------

agent = create_tool_calling_agent(llm, tools, prompt)


# ---------------------------------------------------------------------
# 4) EXECUTOR — runs the loop: think, act, observe, repeat, until the
#    agent is ready to give a final answer. verbose=True prints each
#    step live, which is what you want on screen for the recording.
# ---------------------------------------------------------------------

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


if __name__ == "__main__":
    question = (
        "What is 23 * 17, and how many letters are in the word "
        "'orchestration'?"
    )
    print(f"\n>>> {question}\n")
    result = agent_executor.invoke({"input": question})
    print("\nFINAL ANSWER:", result["output"][0]["text"])
