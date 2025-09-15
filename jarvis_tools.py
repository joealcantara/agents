from langchain.tools import Tool 

from memory_tool import Memory

memory_core = Memory()

memory_retriever_tool = Tool(
        name="MemoryRetriever",
        func=memory_core.retrieve_memory,
        description="Use this to recall facts from past conversations. The input should be a question about something we have discussed before."
    )
