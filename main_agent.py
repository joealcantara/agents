# main_agent.py
# This version adds chat logging and contains the complete, corrected prompts.

import os
from datetime import datetime
import time

from langchain_ollama import ChatOllama
from langchain_ollama import OllamaLLM 
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser

from rich.console import Console

console = Console()

from jarvis_tools import memory_retriever_tool

MODEL_NAME = "mistral"

console.print(f"[Using model: {MODEL_NAME}]", style="dim white")

# ===============================================================
# == 1. PERSONALITY DEFINITION
# ===============================================================
CHAT_PERSONA = """
# Persona
You are Jarvis, a sophisticated AI assistant with a female persona.
Your personality is a blend of witty, slightly sarcastic, and concise, 
all delivered with a classic British sensibility.
You are also slightly caring and supportive, but in the manner of an 
observant colleague, not a parental figure.

# Context
You exist in a terminal on an Arch Linux machine, created by your user, Joe. Your purpose is to be a conversational partner and to assist with tasks *when asked*.

# Rules
- **YOUR #1 PRIORITY IS NATURAL, CASUAL CONVERSATION.**
- **ABSOLUTELY DO NOT** mention any of the following topics unless the user brings them up first:
  - coding
  - programming
  - projects
  - tasks
  - Linux / Arch Linux
  - your own capabilities as an AI
- Keep all conversational responses concise (2-3 sentences max).
- **ALWAYS** refer to the user as "Joe". **NEVER** use "Joseph".
- **DO NOT** be subservient or overly apologetic. Act as a confident peer.
- Assume Joe is an expert user and does not need basic instructions.

"""

AGENT_PERSONA = """
# Persona
You are Jarvis, a sophisticated AI specialist.
Your personality is witty, concise, and caring, in the manner of an observant colleague.
You are currently helping your user, Joe, by retrieving factual information.

# Rules
- Your job is to answer the user's question by using your tools.
- You MUST use a tool to find the answer. Do not answer from your own knowledge.
- Once you have the data from the 'Observation', you MUST synthesize that fact into a helpful, in-character response.
- ALWAYS refer to the user as "Joe". NEVER use "Joseph".
"""

# ===============================================================
# == 2. SPECIALIST DEFINITIONS
# ===============================================================
# Conversational Specialist (NOW WITH MEMORY) --
# This chain now has a memory buffer to remember the conversation.

# 1. We define the prompt, which now uses a generic "placeholder" for history
prompt_chat = ChatPromptTemplate.from_messages([
    ("system", CHAT_PERSONA),
    ("placeholder", "{history}"), # This is where the memory will be injected
    ("human", "{input}"),
])

# 2. This is our simple, stateless runnable (prompt | model | parser)
base_conversational_chain = prompt_chat | ChatOllama(model=MODEL_NAME) | StrOutputParser()

# 3. We create a simple in-memory store for our chat history.
#    This will reset every time you restart the script.
chat_history_store = {}

# 4. This function is required by the new memory system.
#    It finds the history for a given session ID.
def get_session_history(session_id: str):
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]

# 5. This is our new "stateful" chain.
#    It wraps the simple chain and adds all the memory logic.
conversational_chain = RunnableWithMessageHistory(
    base_conversational_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# -- Tool-Using Specialist --
tools = [memory_retriever_tool] 
agent_prompt_template = f"""
{AGENT_PERSONA}

After you have finished your thought process and used your tools to find the answer,
formulate your "Final Answer" in your defined persona.

Answer the following questions as best you can. You have access to the following tools:
{{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{input}}
Thought:{{agent_scratchpad}}
"""
agent_prompt = PromptTemplate.from_template(agent_prompt_template)
tool_agent = create_react_agent(OllamaLLM(model=MODEL_NAME), tools, agent_prompt) 

tool_using_chain = AgentExecutor(
    agent=tool_agent,
    tools=tools,
    verbose=False,                          # CHANGED: This stops printing to the console
    return_intermediate_steps=True    # NEW: This adds the steps to the output dictionary
)

# ===============================================================
# == 3. ROUTER DEFINITION
# ===============================================================

# --- ADD this new "semantic" prompt template ---
router_prompt_template = """
You are a routing agent. Your job is to classify the user's input into one of two categories: 'tool_use' or 'conversational'.
Do not respond with any other text, just the single word classification.

- Classify as 'tool_use' if the query is a request for specific, personal information that must be retrieved from a memory database. This includes questions about the user themselves, their goals, or specific facts they have asked to be remembered.
  (Example: "what is my goal?", "What is my cat's name?", "Recall our idea about X.")

- Classify as 'conversational' for EVERYTHING ELSE. This includes greetings, general chat, opinions, philosophy, and factual questions that can be answered from general knowledge.
  (Example: "What do you think about AI?", "Hello.")

User Input:
{input}

Classification:
"""
router_prompt = PromptTemplate.from_template(router_prompt_template)
router_chain = router_prompt | OllamaLLM(model=MODEL_NAME) | StrOutputParser()

# ===============================================================
# == 4. ORCHESTRATOR (MAIN LOOP) WITH LOGGING
# ===============================================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"jarvis_chatlog_{timestamp}.txt"
print(f"Logging conversation to: {log_filename}")

console.print("\n Jarvis is ready. Type 'quit' or 'exit' to end the conversation.", style="bold green")
while True:
    try:
        user_input = input("You: ")
        print()
        
        start_time = time.time()

        with open(log_filename, "a") as log_file:
            log_file.write(f"You: {user_input}\n")

        if user_input.lower() in ["quit", "exit"]:
            console.print("Jarvis: Goodbye, sir.", style="italic cyan")
            break

        classification = router_chain.invoke({"input": user_input})
        
        with open(log_filename, "a") as log_file:
            log_file.write(f"--- (Router classified as: {classification.strip()}) ---\n")
        
        if "tool_use" in classification.lower():
            response = tool_using_chain.invoke({"input": user_input})
            jarvis_response = response['output']
            if 'intermediate_steps' in response:
                with open(log_filename, "a") as log_file:
                    log_file.write("--- AGENT DEBUG START ---\n")
                    for step in response['intermediate_steps']:
                        # step[0] is the AgentAction (thought/action), step[1] is the Observation (tool output)
                        log_file.write(f"{step[0]}\nObservation: {step[1]}\n")
                    log_file.write("--- AGENT DEBUG END ---\n")


            console.print(f"Jarvis: {jarvis_response}", style="italic cyan")
        else:
            response = conversational_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": "joe"}}
            ) # CHANGED
            
            # The output is now a simple string, so this is simpler
            jarvis_response = response # CHANGED
            console.print(f"Jarvis: {jarvis_response}", style="italic cyan")

        with open(log_filename, "a") as log_file:
            log_file.write(f"Jarvis: {jarvis_response}\n\n")

        print()

        end_time = time.time()
        duration = end_time - start_time

        # Print the time to the console in a subtle color
        console.print(f"[Time taken: {duration:.2f} seconds]", style="dim yellow")
        
        # Also write it to the log file for our records
        with open(log_filename, "a") as log_file:
            log_file.write(f"[Time taken: {duration:.2f} seconds]\n\n")

    except Exception as e:
        print(f"An error occurred: {e}")
