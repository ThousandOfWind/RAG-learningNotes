import logging
import sys
import os
from dotenv import load_dotenv

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.vector_stores.azureaisearch import (
    IndexManagement,
)

load_dotenv()


aoai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
aoai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
aoai_api_version = "2023-05-15"

def get_llm(model:str, deployment:str):
    llm = AzureOpenAI(
        model=model,
        deployment_name=deployment,
        api_key=aoai_api_key,
        azure_endpoint=aoai_endpoint,
        api_version=aoai_api_version,
    )
    return llm

def get_embed_model(model:str, deployment:str):
    embed_model = AzureOpenAIEmbedding(
        model=model,
        deployment_name=deployment,
        api_key=aoai_api_key,
        azure_endpoint=aoai_endpoint,
        api_version=aoai_api_version,
    )
    return embed_model

search_service_api_key = os.getenv('AZURE_SEARCH_API_KEY')
search_service_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
search_service_api_version = "2023-11-01"
credential = AzureKeyCredential(search_service_api_key)


def get_index_client():
    # Use index client to demonstrate creating an index
    index_client = SearchIndexClient(
        endpoint=search_service_endpoint,
        credential=credential,
    )
    return index_client

def get_search_client(index_name:str):
    # Use search client to demonstration using existing index
    search_client = SearchClient(
        endpoint=search_service_endpoint,
        index_name=index_name,
        credential=credential,
    )
    return search_client

def get_vector_store(index_name:str, metadata_fields:dict|None = None):
    """
Field Name	OData Type
id	Edm.String
chunk	Edm.String
embedding	Collection(Edm.Single)
metadata	Edm.String
doc_id	Edm.String
author	Edm.String
theme	Edm.String
director	Edm.String
"""
    vector_store = AzureAISearchVectorStore(
        search_or_index_client=get_index_client(),
        filterable_metadata_field_keys=metadata_fields,
        index_name=index_name,
        index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
        id_field_key="id",
        chunk_field_key="chunk",
        embedding_field_key="embedding",
        embedding_dimensionality=1536,
        metadata_string_field_key="metadata",
        doc_id_field_key="doc_id",
        # https://learn.microsoft.com/en-us/azure/search/index-add-language-analyzers
        language_analyzer="zh-Hans.lucene",
        vector_algorithm_type="exhaustiveKnn",
    )
    return vector_store