# from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import os

# persist_path = os.path.abspath("./petpal_chromadb")
# print("USING CHROMADB VERSION:", __import__("chromadb").__version__)
# print("Persist path:", persist_path)

# embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# collection = Chroma(
#     persist_directory=persist_path,
#     embedding_function=embedding_function,
#     collection_name="pet_memories"
# )

# collection.add_texts(
#     texts=["I like burgers."],
#     ids=["test1"],
#     metadatas=[{"user_id": "43", "pet_id": "35"}]
# )

# results = collection.similarity_search(
#     query="burgers",
#     k=1,
#     filter={"user_id": "43", "pet_id": "35"}
# )
# print("Results:", [doc.page_content for doc in results])
