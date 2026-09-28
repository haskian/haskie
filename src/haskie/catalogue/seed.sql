-- The model catalogue a fresh home starts with: every model the runtimes can load, with its
-- metadata, and every embedding profile. Run once, right after `db.schema_ddl()` (see `db.migrate`), so
-- the database is where the catalogue lives from then on, not this file.
--
-- Parameter counts are the published weights' own totals. `context_tokens` is the longest input
-- the model states it reads; the loaders may cut inputs shorter (`mlx_models`, `gguf_models`,
-- `onnx_rerank`).
-- Each model needs a loader in code: fastembed lists it, or a pin in `embed`, `onnx_rerank`,
-- `mlx_models` or `gguf_models` names it (`tests/catalogue` keeps the two in step).
--
-- `released` and `model_card_url` are the original model's, not a conversion's: an MLX, GGUF or
-- ONNX copy (mlx-community, ggml-org, Xenova, ...) links to the weights it was made from, and is dated by their
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
    -- the official GGUF conversions, on the Apple GPU through llama.cpp (`gguf_models`)
    ('ggml-org/bge-small-en-v1.5-Q8_0-GGUF', 'embedder', 'bge-small on the Apple GPU through llama.cpp, 8-bit (~37 MB).', 33360512, 512, 'English', 'MIT', '2023-09-12', 'https://huggingface.co/BAAI/bge-small-en-v1.5'),
    ('ggml-org/jina-embeddings-v2-base-en-Q8_0-GGUF', 'embedder', 'jina-v2-base on the Apple GPU through llama.cpp, 8-bit; long passages (~146 MB).', 137368320, 8192, 'English', 'Apache-2.0', '2023-09-27', 'https://huggingface.co/jinaai/jina-embeddings-v2-base-en'),
    ('nomic-ai/nomic-embed-text-v1.5-GGUF', 'embedder', 'nomic v1.5 on the Apple GPU through llama.cpp, 16-bit; long passages (~274 MB).', 136731648, 8192, 'English', 'Apache-2.0', '2024-02-10', 'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5'),
    ('ggml-org/bge-m3-Q8_0-GGUF', 'embedder', 'bge-m3 on the Apple GPU through llama.cpp, 8-bit; strong multilingual retrieval (~635 MB).', 567754752, 8192, 'multilingual (100+)', 'MIT', '2024-01-27', 'https://huggingface.co/BAAI/bge-m3'),
    ('Xenova/ms-marco-MiniLM-L-6-v2', 'reranker', 'Small and fast; a good first reranker.', 22714113, 512, 'English', 'Apache-2.0', '2021-04-15', 'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2'),
    ('Xenova/ms-marco-MiniLM-L-12-v2', 'reranker', 'Twice the layers of L-6: a little better, a little slower.', 33360897, 512, 'English', 'Apache-2.0', '2021-04-15', 'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2'),
    ('BAAI/bge-reranker-base', 'reranker', 'Stronger than MiniLM, and much larger.', 278044931, 512, 'English, Chinese', 'MIT', '2023-09-11', 'https://huggingface.co/BAAI/bge-reranker-base'),
    ('jinaai/jina-reranker-v1-turbo-en', 'reranker', 'Distilled for speed; reads long passages.', 37771777, 8192, 'English', 'Apache-2.0', '2024-04-15', 'https://huggingface.co/jinaai/jina-reranker-v1-turbo-en'),
    ('cross-encoder/ettin-reranker-68m-v1', 'reranker', 'Modern and small (2026, ModernBERT-style encoder); reads long passages.', 68144640, 8192, 'English', 'Apache-2.0', '2026-05-15', 'https://huggingface.co/cross-encoder/ettin-reranker-68m-v1'),
    ('mixedbread-ai/mxbai-rerank-xsmall-v1', 'reranker', 'mixedbread''s smallest reranker (2024), strong for its size.', 70830337, 512, 'English', 'Apache-2.0', '2024-02-29', 'https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1'),
    ('mixedbread-ai/mxbai-rerank-base-v1', 'reranker', 'xsmall''s larger sibling: more accurate, about 2x slower (~740 MB).', 184422913, 512, 'English', 'Apache-2.0', '2024-02-29', 'https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1'),
    ('jinaai/jina-reranker-v2-base-multilingual', 'reranker', 'Multilingual cross-encoder.', 278437633, 1024, 'multilingual', 'CC BY-NC 4.0 (non-commercial)', '2024-06-19', 'https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual'),
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
insert or ignore into embedding_profiles (profile, model, dims, description, query_prefix, document_prefix, matryoshka_layer_norm, duplicate_chunk, duplicate_passage, weak_match, answered_match, same_topic)
values
    ('compact', 'BAAI/bge-small-en-v1.5', 384, null, '', '', null, 0.92, 0.95, 0.67, 0.775, 0.70),
    ('balanced', 'BAAI/bge-base-en-v1.5', 768, null, '', '', null, 0.92, 0.95, null, null, null),
    ('gte-base', 'thenlper/gte-base', 768, null, '', '', null, null, null, null, null, null),
    ('arctic-m', 'snowflake/snowflake-arctic-embed-m', 768, null, 'Represent this sentence for searching relevant passages: ', '', null, null, null, 0.27, null, 0.54),
    ('nomic-v1.5', 'nomic-ai/nomic-embed-text-v1.5', 768, null, 'search_query: ', 'search_document: ', null, null, null, null, null, null),
    ('jina-v2-small', 'jinaai/jina-embeddings-v2-small-en', 512, null, '', '', null, null, null, null, null, null),
    -- nomic v1.5 cut to 512 of its 768: 61.96 on MTEB against 62.28 whole, by its card
    ('nomic-v1.5-512','nomic-ai/nomic-embed-text-v1.5', 512, 'nomic v1.5 with its vectors cut to 512 (Matryoshka): a third less index for a small loss (~520 MB).', 'search_query: ', 'search_document: ', 1, null, null, null, null, null),
    ('jina-v2-base', 'jinaai/jina-embeddings-v2-base-en', 768, null, '', '', null, null, null, null, null, null),
    ('modernbert-mlx', 'mlx-community/nomicai-modernbert-embed-base-bf16', 768, null, 'search_query: ', 'search_document: ', null, null, null, null, null, null),
    ('jina-v5-nano-mlx', 'jinaai/jina-embeddings-v5-text-nano-retrieval-mlx', 768, null, '', '', null, null, null, null, null, null),
    ('quality', 'BAAI/bge-large-en-v1.5', 1024, null, '', '', null, 0.92, 0.95, null, null, null),
    ('multilingual', 'intfloat/multilingual-e5-large', 1024, null, 'query: ', 'passage: ', null, 0.97, 0.98, null, null, null),
    ('bge-m3', 'BAAI/bge-m3', 1024, null, '', '', null, null, null, null, null, null),
    -- each GGUF profile as its ONNX twin: the same sizes, prefixes and duplicate cosines
    ('bge-small-gguf', 'ggml-org/bge-small-en-v1.5-Q8_0-GGUF', 384, null, '', '', null, 0.92, 0.95, 0.67, 0.775, 0.70),
    ('jina-v2-base-gguf', 'ggml-org/jina-embeddings-v2-base-en-Q8_0-GGUF', 768, null, '', '', null, null, null, null, null, null),
    ('nomic-v1.5-gguf', 'nomic-ai/nomic-embed-text-v1.5-GGUF', 768, null, 'search_query: ', 'search_document: ', null, null, null, null, null, null),
    ('bge-m3-gguf', 'ggml-org/bge-m3-Q8_0-GGUF', 1024, null, '', '', null, null, null, null, null, null),
    ('jina-v3', 'jinaai/jina-embeddings-v3', 1024, null, '', '', null, null, null, null, null, null);

-- Each reranker's floor and score curve (`reranker_calibration`). Uncalibrated until measured:
-- a floor of 0.05 (logit about -3), judged on one book with MiniLM-L-6, and the identity curve
-- (beta 1, 1). `mise run calibrate-rerankers` measures both on borderline pairs of your own
-- collections, the way Cohere sets a relevance threshold: the average score of 30 to 50 pairs a
-- person judged borderline relevant.
insert or ignore into reranker_calibration (model, floor, beta_a, beta_b, source)
values
    ('Xenova/ms-marco-MiniLM-L-6-v2', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('Xenova/ms-marco-MiniLM-L-12-v2', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('BAAI/bge-reranker-base', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('jinaai/jina-reranker-v1-turbo-en', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('cross-encoder/ettin-reranker-68m-v1', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('mixedbread-ai/mxbai-rerank-xsmall-v1', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('mixedbread-ai/mxbai-rerank-base-v1', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('jinaai/jina-reranker-v2-base-multilingual', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('soichisumi/bge-reranker-v2-m3-mlx-affine8', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('afanjul/gte-reranker-modernbert-base-mlx', 0.05, 1.0, 1.0, 'uncalibrated');

-- `weak_match`, `answered_match` and `same_topic` in the profiles above are the cosines the Gaps
-- page judges searches by (`search.gaps`), measured with `mise run evaluate-gaps`, not guessed. A
-- best cosine under `weak_match` is a gap; from it to `answered_match`, borderline (maybe
-- answered); over it, an answer. A reranked search is judged by the reranker's floor instead,
-- since the reranker reads query and passage together. A profile without `weak_match` gives no
-- cosine verdict; without `answered_match`, no borderline band; without `same_topic`, gaps are
-- grouped by their words. Each bar is set so that no answered question is flagged: a false gap
-- costs the curator's trust, a missed one only waits for the next search.
--
-- Two shelves (tests/gapeval), each pinned to a commit: the Rust book (1,418 chunks at the default
-- chunk settings; 45 answered questions, 40 unanswered: 30 near its topics, 10 far) and four of
-- haskie's docs (80 chunks; 12 answered, 5 unanswered). `answered_match` must hold every unanswered question the
-- low bar misses while flagging at most 15% of the answered ones; else a profile has no band.
-- `same_topic` sits over the highest cosine between two queries of different topics (12 topics,
-- 3 phrasings each), so two topics never merge.
--
-- - compact: the lowest answered best cosine is 0.673 (haskie docs; the Rust book's is 0.751),
--   so 0.67; it catches 30 of 45 unanswered. The highest unanswered is 0.763, so the band ends at
--   0.775: it holds the other 15, with 5 of 57 answered (9%). Topics: other pairs <= 0.692, so
--   0.70, joining 89% of same-topic pairs. An edit to the docs alone moved their lowest answered
--   cosine from 0.698 to 0.673: a bar holds for the text it was measured on.
-- - arctic-m: answered down to 0.278 (haskie docs; 0.386 on the Rust book), unanswered up to
--   0.464: they overlap, so 0.27 catches 13 of 45, and no band passes (holding all 32 others
--   would flag 40% of answered). Topics: other pairs <= 0.525, so 0.54.
-- - bge-small-gguf borrows compact's values, as it borrows its duplicate cosines.
-- - Measured against the rerankers' floors above: MiniLM-L-6 at 0.05 flags no answered question
--   and catches 36 of 45 (compact pools); mxbai-rerank-xsmall at 0.05 flags none and catches 24
--   of 45 (arctic-m pools).
