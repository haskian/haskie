-- The model catalogue a fresh home starts with: every model the runtimes can load, with its
-- metadata, and every embedding profile. Run once, right after `db.SCHEMA` (see `db.migrate`), so
-- the database is where the catalogue lives from then on, not this file.
--
-- Parameter counts are the published weights' own totals. `context_tokens` is the longest input
-- the model states it reads; the loaders may cut inputs shorter (`mlx_models`, `onnx_rerank`).
-- Each model needs a loader in code: fastembed lists it, or a pin in `embed`, `onnx_rerank` or
-- `mlx_models` names it (`tests/catalogue` keeps the two in step).
--
-- `released` and `model_card_url` are the original model's, not a conversion's: an MLX or ONNX
-- copy (mlx-community, Xenova, ...) links to the weights it was made from, and is dated by their
-- first commit on the Hugging Face Hub. The Hub's `createdAt` is no substitute: repositories
-- older than March 2022 all show 2022-03-02, the day it began recording it. Which runtime loads a
-- model and which devices run it are not here: the loaders decide both (`indexing.hardware`).

insert or ignore into models (
    name, kind, description, parameters, context_tokens, languages, license, released, model_card_url
)
values
    ('BAAI/bge-small-en-v1.5', 'embedder', 'Small and fast; a good default (~130 MB).', 33360512, 512, 'English', 'MIT', '2023-09-12', 'https://huggingface.co/BAAI/bge-small-en-v1.5'),
    ('BAAI/bge-base-en-v1.5', 'embedder', 'Between bge-small and bge-large: 768-dimension vectors at a third of bge-large''s size (~210 MB).', 109482752, 512, 'English', 'MIT', '2023-09-11', 'https://huggingface.co/BAAI/bge-base-en-v1.5'),
    ('thenlper/gte-base', 'embedder', 'General text embeddings from Alibaba; no prefixes needed (~440 MB).', 109482752, 512, 'English', 'MIT', '2023-07-27', 'https://huggingface.co/thenlper/gte-base'),
    ('snowflake/snowflake-arctic-embed-m', 'embedder', 'Snowflake''s retrieval model, tuned for search queries (~430 MB).', 108891648, 512, 'English', 'Apache-2.0', '2024-04-11', 'https://huggingface.co/Snowflake/snowflake-arctic-embed-m'),
    ('nomic-ai/nomic-embed-text-v1.5', 'embedder', 'Long passages, open training data (~520 MB).', 136731648, 8192, 'English', 'Apache-2.0', '2024-02-10', 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5'),
    ('jinaai/jina-embeddings-v2-small-en', 'embedder', 'Small, with long passages and no prefixes (~120 MB).', 32690688, 8192, 'English', 'Apache-2.0', '2023-09-27', 'https://huggingface.co/jinaai/jina-embeddings-v2-small-en'),
    ('jinaai/jina-embeddings-v2-base-en', 'embedder', 'Long passages without prefixes (~520 MB).', 137368320, 8192, 'English', 'Apache-2.0', '2023-09-27', 'https://huggingface.co/jinaai/jina-embeddings-v2-base-en'),
    ('mlx-community/nomicai-modernbert-embed-base-bf16', 'embedder', 'nomic''s ModernBERT embedder on the Apple GPU; long passages (~300 MB).', 149014272, 8192, 'English', 'Apache-2.0', '2024-12-29', 'https://huggingface.co/nomic-ai/modernbert-embed-base'),
    ('jinaai/jina-embeddings-v5-text-nano-retrieval-mlx', 'embedder', 'Jina''s newest small embedder, for retrieval, on the Apple GPU (~420 MB). Adds its own query and document prefixes.', 211766016, 8192, 'multilingual', 'CC BY-NC 4.0 (non-commercial)', '2026-02-10', 'https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval'),
    ('BAAI/bge-large-en-v1.5', 'embedder', 'Better recall than bge-small, slower (~1.3 GB).', 335142400, 512, 'English', 'MIT', '2023-09-12', 'https://huggingface.co/BAAI/bge-large-en-v1.5'),
    ('intfloat/multilingual-e5-large', 'embedder', 'Multilingual (~2.2 GB).', 559890946, 512, 'multilingual (~94)', 'MIT', '2023-06-30', 'https://huggingface.co/intfloat/multilingual-e5-large'),
    -- its weights ship as a .bin, which the Hub does not count: XLM-RoBERTa large with 8194
    -- positions, as bge-reranker-v2-m3's 567755777 is with its head in place of the pooler
    ('BAAI/bge-m3', 'embedder', 'Strong multilingual retrieval; dense vectors only (~2.3 GB).', 567754752, 8192, 'multilingual (100+)', 'MIT', '2024-01-27', 'https://huggingface.co/BAAI/bge-m3'),
    ('jinaai/jina-embeddings-v3', 'embedder', 'Multilingual, with task adapters for queries and passages (~2.3 GB).', 572310396, 8192, 'multilingual (~94)', 'CC BY-NC 4.0 (non-commercial)', '2024-09-05', 'https://huggingface.co/jinaai/jina-embeddings-v3'),
    ('Xenova/ms-marco-MiniLM-L-6-v2', 'reranker', 'Small and fast; a good first reranker.', 22714113, 512, 'English', 'Apache-2.0', '2021-04-15', 'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2'),
    ('Xenova/ms-marco-MiniLM-L-12-v2', 'reranker', 'Twice the layers of L-6: a little better, a little slower.', 33360897, 512, 'English', 'Apache-2.0', '2021-04-15', 'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2'),
    ('BAAI/bge-reranker-base', 'reranker', 'Stronger than MiniLM, and much larger.', 278044931, 512, 'English, Chinese', 'MIT', '2023-09-11', 'https://huggingface.co/BAAI/bge-reranker-base'),
    ('jinaai/jina-reranker-v1-turbo-en', 'reranker', 'Distilled for speed; reads long passages.', 37771777, 8192, 'English', 'Apache-2.0', '2024-04-15', 'https://huggingface.co/jinaai/jina-reranker-v1-turbo-en'),
    ('cross-encoder/ettin-reranker-68m-v1', 'reranker', 'Modern and small (2026, ModernBERT-style encoder); reads long passages.', 68144640, 8192, 'English', 'Apache-2.0', '2026-05-15', 'https://huggingface.co/cross-encoder/ettin-reranker-68m-v1'),
    ('mixedbread-ai/mxbai-rerank-xsmall-v1', 'reranker', 'mixedbread''s smallest reranker (2024), strong for its size.', 70830337, 512, 'English', 'Apache-2.0', '2024-02-29', 'https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1'),
    ('jinaai/jina-reranker-v2-base-multilingual', 'reranker', 'Multilingual cross-encoder.', 278437633, 1024, 'multilingual', 'CC BY-NC 4.0 (non-commercial)', '2024-06-19', 'https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual'),
    ('jinaai/jina-reranker-v3-mlx', 'reranker', 'Listwise: reads every candidate together and ranks them against each other.', 596836352, 131072, 'multilingual', 'CC BY-NC 4.0 (non-commercial)', '2025-09-18', 'https://huggingface.co/jinaai/jina-reranker-v3'),
    ('jinaai/jina-reranker-v3.5-mlx', 'reranker', 'Jina''s newest listwise reranker, v3''s successor.', 596836352, 131072, 'multilingual', 'CC BY-NC 4.0 (non-commercial)', '2026-07-14', 'https://huggingface.co/jinaai/jina-reranker-v3.5'),
    ('soichisumi/bge-reranker-v2-m3-mlx-affine8', 'reranker', 'BAAI''s multilingual reranker, 8-bit (near lossless, 607 MB).', 567755777, 8192, 'multilingual (100+)', 'Apache-2.0', '2024-03-15', 'https://huggingface.co/BAAI/bge-reranker-v2-m3'),
    ('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'reranker', 'Multilingual MiniLM trained on mMARCO; the fastest multilingual reranker here.', 117641603, 512, 'multilingual (mMARCO: 14 languages)', 'Apache-2.0', '2022-06-01', 'https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'),
    ('afanjul/gte-reranker-modernbert-base-mlx', 'reranker', 'Fast modern cross-encoder; reads long passages.', 149605633, 8192, 'English', 'Apache-2.0', '2025-01-20', 'https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base');

-- The `duplicate_*` cosines are placeholders, not calibrated. No source gives a portable value:
-- SemDeDup tunes its threshold "for each dataset manually" [1], and the BGE card says to pick one
-- from your own score distribution [3]. Calibrate on ~50 labelled pairs per model and per level,
-- keeping precision near 1, so a wrong value folds too little rather than too much. A profile
-- without them folds its search results by words alone (`search.collapse`).
--
-- - bge chunk 0.92: just under the 0.93 NeMo Curator uses in its examples (its default is 0.99,
--   with another model) [2]. Measured once, with bge-small: the same paragraph under two different
--   heading paths scored 0.927, which is why a copy the words find folds anyway
--   (`search.collapse.Embedded`).
-- - passage above chunk: a mean vector is smoother than its chunks, so means of related spans sit
--   closer together (inference, redundancy-diversity-coverage.md §3.4).
-- - e5 above bge: unrelated pairs average a raw cosine of 0.707 for multilingual-e5-large against
--   0.308 for bge-large [4], and the e5 card puts its scores "around 0.7 to 1.0" [5]. The size of
--   the step (0.97, 0.98) is judgement, not a formula. bge-small and bge-base were not measured
--   in [4] and borrow bge-large's values.
--
-- [1] Abbas et al. 2023, SemDeDup, https://arxiv.org/abs/2303.09540
-- [2] NVIDIA NeMo Curator, semantic dedup, https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/semdedup.html
-- [3] https://huggingface.co/BAAI/bge-large-en-v1.5 (FAQ on similarity scores)
-- [4] Parupudi 2026, Table 1 (preprint, single author), https://arxiv.org/abs/2606.29571
-- [5] https://huggingface.co/intfloat/multilingual-e5-large (FAQ)
--
-- `matryoshka_layer_norm` set means the vectors are cut to `dims` (Matryoshka Representation
-- Learning); 1 is nomic's recipe, a layer norm over the whole vector before the cut.
insert or ignore into embedding_profiles (profile, model, dims, description, query_prefix, document_prefix, matryoshka_layer_norm, duplicate_chunk, duplicate_passage)
values
    ('compact', 'BAAI/bge-small-en-v1.5', 384, null, '', '', null, 0.92, 0.95),
    ('balanced', 'BAAI/bge-base-en-v1.5', 768, null, '', '', null, 0.92, 0.95),
    ('gte-base', 'thenlper/gte-base', 768, null, '', '', null, null, null),
    ('arctic-m', 'snowflake/snowflake-arctic-embed-m', 768, null, 'Represent this sentence for searching relevant passages: ', '', null, null, null),
    ('nomic-v1.5', 'nomic-ai/nomic-embed-text-v1.5', 768, null, 'search_query: ', 'search_document: ', null, null, null),
    ('jina-v2-small', 'jinaai/jina-embeddings-v2-small-en', 512, null, '', '', null, null, null),
    -- nomic v1.5 cut to 512 of its 768: 61.96 on MTEB against 62.28 whole, by its card
    ('nomic-v1.5-512','nomic-ai/nomic-embed-text-v1.5', 512, 'nomic v1.5 with its vectors cut to 512 (Matryoshka): a third less index for a small loss (~520 MB).', 'search_query: ', 'search_document: ', 1, null, null),
    ('jina-v2-base', 'jinaai/jina-embeddings-v2-base-en', 768, null, '', '', null, null, null),
    ('modernbert-mlx', 'mlx-community/nomicai-modernbert-embed-base-bf16', 768, null, 'search_query: ', 'search_document: ', null, null, null),
    ('jina-v5-nano-mlx', 'jinaai/jina-embeddings-v5-text-nano-retrieval-mlx', 768, null, '', '', null, null, null),
    ('quality', 'BAAI/bge-large-en-v1.5', 1024, null, '', '', null, 0.92, 0.95),
    ('multilingual', 'intfloat/multilingual-e5-large', 1024, null, 'query: ', 'passage: ', null, 0.97, 0.98),
    ('bge-m3', 'BAAI/bge-m3', 1024, null, '', '', null, null, null),
    ('jina-v3', 'jinaai/jina-embeddings-v3', 1024, null, '', '', null, null, null);
