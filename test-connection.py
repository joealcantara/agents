from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="mistral")

print("Sending a prompt to the Mistral model...")

response = llm.invoke("Why is the sky blue?")

print("\n, Model Response:")
print(response)
