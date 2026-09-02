from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"


# Load embeddings and vector store
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

retriever = db.as_retriever(search_kwargs={"k": 5})

query = "Who started tesla company?"

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")


print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

combined_input = f"""Based on the follwong documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Answer using only the information in the documents above. The documents may imply an answer without
stating it in the exact same words as the question (e.g. a date, name, or event described in different
phrasing still counts as the answer) -- read closely, compare any dates given, and make reasonable
inferences from what is stated. Only say you are unable to answer if the documents truly contain nothing
relevant.
"""

model = OllamaLLM(model="llama3.1:8b")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]
result = model.invoke(messages)
print("\n--- Generated Response ---")
# print("Full result")
#print(result)
print("Content only:")
print(result)



