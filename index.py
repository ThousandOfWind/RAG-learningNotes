from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.settings import Settings


def get_index(vector_store, llm, embed_model):
    Settings.llm = llm
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model)
    return index


def build_index(vector_store, llm, embed_model, splitter, documents):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = splitter
    

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, transformations=[splitter.get_nodes_from_documents]
    )
    return index


