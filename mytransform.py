from llama_index.core.extractors import BaseExtractor

def get_chapter_title(node_metadata: dict):
    if node_metadata['Header_1']:
        ind = node_metadata['Header_1'].find("{")
        if ind > 0:
            return node_metadata['Header_1'][:ind]
    return ""

class MyExtractor(BaseExtractor):
    async def aextract(self, nodes):
        metadata_list = [
            {
                "chapter_title": get_chapter_title(node.metadata)
            }
            for node in nodes
        ]
        return metadata_list

import re
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent


class ImageCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.text = re.sub(r"::: \{\.duokan-image-single\}\n!\[image\]\(Images\/[0-9]+\.jpeg\)\n\n.*\n:::", "", node.text)
        return nodes