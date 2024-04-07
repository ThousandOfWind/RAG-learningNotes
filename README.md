# RAG learning Note


## RAG Workflow and Pain Point

### Ingestion
#### Classic

[document reader] -> [embedding] -> [vector db]

#### Pain points and Solution

* long document -> splitter
* low quality -> data clean
* lack keywords -> attach metadata
* different from user query -> predict question
* fail in domain knowledge -> finetune embedding model

#### Optimal

[document reader] -> [node preprocessor]-> [embedding] -> [vector db]

### Answer Question 
#### Classic
[question] -> [retrieve] -> [synthesis answer]

**Pain points and Solution**
* question: rewrite question
* complex question: sub-question / multi-step query
* asymmetric query and context:adapter / finetune embedding
* loss in middle: rerank /reorder
* context to long: filter / compression /tree summrize
* cite the source: special prompt when synthesis

#### Optimal
[question] -> {loop: [question transform] -> [adapter] -> [retrieve] -> [post retrieve processor] -> [synthesis answer]}


## Reference:
* [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997)
* [LlamaIndex](https://docs.llamaindex.ai/en/stable/)