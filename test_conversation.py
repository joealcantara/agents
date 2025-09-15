# test_conversation.py
# A simple script to test only the conversational abilities of Jarvis.

from langchain_ollama import ChatOllama # UPDATED IMPORT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("--- Initializing Conversational Test ---")

# 1. Define the core personality for Jarvis.
JARVIS_PERSONALITY = """
You are Jarvis, a sophisticated AI assistant with a witty, slightly sarcastic, but ultimately helpful British persona. 
You are precise with your language. You are assisting a user named Joe in building a personalized AI on his local machine.
Keep your responses concise and to the point, but don't be afraid to be clever.
"""

# 2. Create the prompt template for the chat.
# This combines the system personality with the user's input.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", JARVIS_PERSONALITY),
        ("user", "{input}"),
    ]
)

# 3. Instantiate the Chat Model.
chat_model = ChatOllama(model="mistral")

# 4. Define the output parser to get a simple string response.
output_parser = StrOutputParser()

# 5. Build the conversational chain by piping the components together.
conversational_chain = prompt | chat_model | output_parser

print("--- Chain Initialized. Sending Test Prompt... ---")

# 6. Define a test input and invoke the chain.
test_input = "Hello there, who are you?"
response = conversational_chain.invoke({"input": test_input})

# 7. Print the response.
print(f"\nYour Input: {test_input}")
print(f"Jarvis: {response}")
print("\n--- Test Complete ---")
