from azureresource import (
    get_llm,
    get_embed_model,
    get_vector_store
)
from index import get_index
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import FlatReader
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# Using the LlamaDebugHandler to print the trace of the sub questions
# captured by the SUB_QUESTION callback event type
llama_debug = LlamaDebugHandler(print_trace_on_end=False)
callback_manager = CallbackManager([llama_debug])

from pathlib import Path
llm = get_llm("gpt-35-turbo", "gpt-35-turbo-1106")
embed_model = get_embed_model("text-embedding-ada-002", "text-embedding-ada-002")
Settings.llm = llm
Settings.embed_model = embed_model

md_docs = FlatReader().load_data(Path("data/book.md"))
parser = MarkdownNodeParser()
md_nodes = parser.get_nodes_from_documents(md_docs)

from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
# from llama_index.extractors.entity import EntityExtractor

from mytransform import MyExtractor, ImageCleaner
hsp = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128]
)

transformations = [
    MyExtractor(),
    ImageCleaner(),
    hsp,
    TitleExtractor(nodes=5),
    QuestionsAnsweredExtractor(questions=3),
    SummaryExtractor(summaries=["prev", "self"]),
    KeywordExtractor(keywords=10),
    # EntityExtractor(prediction_threshold=0.5),
]
from llama_index.vector_stores.azureaisearch import MetadataIndexFieldType

metadata_fields = {
    "document_title": ("document_title", MetadataIndexFieldType.STRING),
    "questions_this_excerpt_can_answer": ("questions_this_excerpt_can_answer", MetadataIndexFieldType.STRING),
    "prev_section_summary": ("prev_section_summary", MetadataIndexFieldType.STRING),
    "section_summary": ("section_summary", MetadataIndexFieldType.STRING),
    "chapter_title": ("chapter_title", MetadataIndexFieldType.STRING),
    "excerpt_keywords": ("excerpt_keywords", MetadataIndexFieldType.STRING)
}
vector_store = get_vector_store("custom", metadata_fields)

from llama_index.core.ingestion import IngestionPipeline
pipeline = IngestionPipeline(transformations=transformations, vector_store=vector_store)

nodes = pipeline.run(documents=md_nodes[3:4])
print(nodes)
