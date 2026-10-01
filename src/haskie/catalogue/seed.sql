-- The model catalogue a fresh home starts with: every model the runtimes can load, with its
-- metadata, and every embedding profile. Run once, right after `db.schema_ddl()` (see `db.migrate`), so
-- the database is where the catalogue lives from then on, not this file. An edit here reaches only
-- a home created after it: an existing home keeps the catalogue it was seeded with.
--
-- Parameter counts are the published weights' own totals. `context_tokens` is the longest input
-- the model states it reads; the loaders may cut inputs shorter (`onnx_models`, `mlx_models`,
-- `gguf_models`).
-- Each model needs a loader in code: a pin in `onnx_models`, `mlx_models` or `gguf_models` names
-- it (`tests/catalogue` keeps the two in step). Every ONNX model here matches sentence-transformers
-- on the original weights (worst cosine 0.9996, reranker scores to 1e-5), and runs on the Apple
-- GPU through WebGPU. An MLX or GGUF model is one of them run another way, faster on that GPU and
-- matching its ONNX export to a worst cosine of 0.9999; each module says which were left out.
--
-- A rank in a description is among the small models (under 200M parameters) compared in October
-- 2026, by nDCG@10: the embedders on RTEB, MTEB(eng, v2) and MTEB(Multilingual, v2) retrieval and
-- RAR-b, from the MTEB results repository [6]; the rerankers on MTEB(eng, v2) retrieval reranking,
-- from the Ettin release [7], and on long documents (MLDR), from IBM's reranker card [8].
--
-- `released` and `model_card_url` are the original model's, not a conversion's: an MLX, GGUF or
-- ONNX copy (onnx-community, mradermacher, `:mlx`, ...) links to the weights it was made from, and is dated by their
-- first commit on the Hugging Face Hub. The Hub's `createdAt` is no substitute: repositories
-- older than March 2022 all show 2022-03-02, the day it began recording it. Which runtime loads a
-- model and which devices run it are not here: the loaders decide both (`indexing.hardware`).

insert or ignore into models (
    name, kind, description, parameters, context_tokens, languages, license, released, model_card_url
)
values
    -- bekko and granite R2 are ModernBERT encoders whose vocabulary table holds most of their
    -- weights: bekko-a8m runs 7.7M of its 106M per token, a25m 24.9M, granite-97m 28.3M
    ('hotchpotch/bekko-embedding-v1-a8m', 'embedder', 'The fastest multilingual embedder here: 7.7M active parameters, #4 on RTEB (~130 MB).', 105975168, 8192, 'multilingual (100+)', 'MIT', '2026-07-07', 'https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m'),
    ('hotchpotch/bekko-embedding-v1-a25m', 'embedder', 'Small and multilingual: #1 on RTEB, realistic enterprise retrieval, at 24.9M active parameters (~200 MB).', 123234432, 8192, 'multilingual (100+)', 'MIT', '2026-07-19', 'https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m'),
    ('ibm-granite/granite-embedding-97m-multilingual-r2', 'embedder', 'The best all-round small multilingual embedder: #1 on multilingual and reasoning retrieval; a good default (~390 MB).', 97441152, 32768, 'multilingual (200+, 52 enhanced)', 'Apache-2.0', '2026-04-20', 'https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2'),
    ('onnx-community/granite-embedding-small-english-r2-ONNX', 'embedder', 'Small and English only: #3 on English retrieval at a third of granite-english''s size (~195 MB).', 47662464, 8192, 'English', 'Apache-2.0', '2025-07-17', 'https://huggingface.co/ibm-granite/granite-embedding-small-english-r2'),
    ('sirasagi62/granite-embedding-english-r2-ONNX', 'embedder', 'The best small English embedder: #1 on English retrieval (~600 MB).', 149014272, 8192, 'English', 'Apache-2.0', '2025-07-17', 'https://huggingface.co/ibm-granite/granite-embedding-english-r2'),
    ('intfloat/e5-base-v2', 'embedder', 'Older, but #2 on reasoning retrieval: questions that need a step of thought (~440 MB).', 109482752, 512, 'English', 'MIT', '2023-05-19', 'https://huggingface.co/intfloat/e5-base-v2'),
    -- a pruned Qwen3: its config holds 40960 positions, and its card states no limit
    ('onnx-community/F2LLM-v2-160M-ONNX', 'embedder', 'Multilingual, #2 on RTEB; reads each query after a fixed instruction (~640 MB).', 159185024, 40960, 'multilingual (200+)', 'Apache-2.0', '2026-03-09', 'https://huggingface.co/codefuse-ai/F2LLM-v2-160M'),
    -- on the Apple GPU only: MLX from the original weights (`:mlx`), GGUF on llama.cpp's Metal.
    -- Times are 128 real chunks against the same model on WebGPU (M4 Pro).
    ('hotchpotch/bekko-embedding-v1-a25m:mlx', 'embedder', 'bekko-a25m on the Apple GPU through MLX: about 2.4x faster than on WebGPU.', 123234432, 8192, 'multilingual (100+)', 'MIT', '2026-07-19', 'https://huggingface.co/hotchpotch/bekko-embedding-v1-a25m'),
    ('intfloat/e5-base-v2:mlx', 'embedder', 'e5-base-v2 on the Apple GPU through MLX: about 1.5-2x faster than on WebGPU.', 109482752, 512, 'English', 'MIT', '2023-05-19', 'https://huggingface.co/intfloat/e5-base-v2'),
    ('codefuse-ai/F2LLM-v2-160M:mlx', 'embedder', 'F2LLM-v2-160M on the Apple GPU through MLX: about 2-3x faster than on WebGPU.', 159185024, 40960, 'multilingual (200+)', 'Apache-2.0', '2026-03-09', 'https://huggingface.co/codefuse-ai/F2LLM-v2-160M'),
    ('mradermacher/granite-embedding-english-r2-GGUF', 'embedder', 'granite-english on the Apple GPU through llama.cpp, 16-bit: about 2.5x faster than on WebGPU (~300 MB).', 149014272, 8192, 'English', 'Apache-2.0', '2025-07-17', 'https://huggingface.co/ibm-granite/granite-embedding-english-r2'),
    ('ChristianAzinn/e5-base-v2-gguf', 'embedder', 'e5-base-v2 on the Apple GPU through llama.cpp, 16-bit: about 3x faster than on WebGPU (~220 MB).', 109482752, 512, 'English', 'MIT', '2023-05-19', 'https://huggingface.co/intfloat/e5-base-v2'),
    ('mradermacher/F2LLM-v2-160M-GGUF', 'embedder', 'F2LLM-v2-160M on the Apple GPU through llama.cpp, 16-bit: about 2x faster than on WebGPU (~320 MB).', 159185024, 40960, 'multilingual (200+)', 'Apache-2.0', '2026-03-09', 'https://huggingface.co/codefuse-ai/F2LLM-v2-160M'),
    -- the map's reranker (`DEFAULT_MAP_RERANKER`), measured for maps before this catalogue
    ('cross-encoder/ms-marco-MiniLM-L2-v2', 'reranker', 'The smallest MiniLM: two layers, about 2.5x faster than L-6 on a map, for 4 points of MRR; the default for maps.', 15616257, 512, 'English', 'Apache-2.0', '2021-04-15', 'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L2-v2'),
    -- each Ettin's own weights; its scoring head adds well under 1%
    ('cross-encoder/ettin-reranker-17m-v1', 'reranker', 'The fastest reranker here, and better than the MiniLM rerankers it replaces (~67 MB).', 16797440, 8192, 'English', 'Apache-2.0', '2026-05-15', 'https://huggingface.co/cross-encoder/ettin-reranker-17m-v1'),
    ('cross-encoder/ettin-reranker-32m-v1', 'reranker', 'Small and accurate: beats rerankers 17 times its size, about 3x slower than ettin-17m; a good default (~128 MB).', 31883136, 8192, 'English', 'Apache-2.0', '2026-05-15', 'https://huggingface.co/cross-encoder/ettin-reranker-32m-v1'),
    ('cross-encoder/ettin-reranker-68m-v1', 'reranker', 'Level with Qwen3-Reranker-0.6B at a ninth of its size (~273 MB).', 68144640, 8192, 'English', 'Apache-2.0', '2026-05-15', 'https://huggingface.co/cross-encoder/ettin-reranker-68m-v1'),
    ('cross-encoder/ettin-reranker-150m-v1', 'reranker', 'The most accurate small reranker; slow on the CPU (~600 MB).', 149014272, 8192, 'English', 'Apache-2.0', '2026-05-15', 'https://huggingface.co/cross-encoder/ettin-reranker-150m-v1'),
    ('Alibaba-NLP/gte-reranker-modernbert-base', 'reranker', 'The best measured on long documents (~600 MB).', 149605633, 8192, 'English', 'Apache-2.0', '2025-01-20', 'https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base'),
    ('jrc2139/granite-embedding-reranker-english-r2-ONNX', 'reranker', 'IBM''s reranker, trained without MS MARCO and its non-commercial license (~600 MB).', 149605633, 8192, 'English', 'Apache-2.0', '2025-08-04', 'https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2');

-- The `duplicate_*` cosines are the raw cosines over which two search results count as one point
-- (`search.collapse`). No profile has them yet: no source gives a portable value. SemDeDup tunes
-- its threshold "for each dataset manually" [1], NeMo Curator's examples use 0.93 with its own
-- model [2], and the BGE card says to pick one from your own score distribution [3]. Unrelated
-- pairs sit at very different cosines from model to model (0.707 on average for
-- multilingual-e5-large against 0.308 for bge-large [4]; e5 puts its scores "around 0.7 to 1.0"
-- [5]), so a value never carries over. Calibrate on ~50 labelled pairs per model and per level,
-- keeping precision near 1, so a wrong value folds too little rather than too much. Until then a
-- profile folds its search results by words alone.
--
-- [1] Abbas et al. 2023, SemDeDup, https://arxiv.org/abs/2303.09540
-- [2] NVIDIA NeMo Curator, semantic dedup, https://docs.nvidia.com/nemo/curator/latest/curate-text/process-data/deduplication/semdedup.html
-- [3] https://huggingface.co/BAAI/bge-large-en-v1.5 (FAQ on similarity scores)
-- [4] Parupudi 2026, Table 1 (preprint, single author), https://arxiv.org/abs/2606.29571
-- [5] https://huggingface.co/intfloat/multilingual-e5-large (FAQ)
-- [6] https://github.com/embeddings-benchmark/results
-- [7] Ettin rerankers, Hugging Face blog, 19 May 2026, https://huggingface.co/cross-encoder/ettin-reranker-150m-v1
-- [8] https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2
--
-- Prefixes are what each model's card asks for: e5 reads "query: " and "passage: ", F2LLM-v2 an
-- instruction ahead of a query and nothing ahead of a document; bekko, granite and Ettin read raw
-- `matryoshka` 1 means the vectors are cut to `dims` (Matryoshka Representation Learning),
-- then normalized again.
insert or ignore into embedding_profiles (profile, model, dims, description, query_prefix, document_prefix, matryoshka, duplicate_chunk, duplicate_passage, weak_match, answered_match, same_topic)
values
    ('bekko-a8m', 'hotchpotch/bekko-embedding-v1-a8m', 384, null, '', '', 0, null, null, null, null, null),
    ('bekko-a25m', 'hotchpotch/bekko-embedding-v1-a25m', 384, null, '', '', 0, null, null, null, null, null),
    -- 256 of its 384 cost 1.4% on HAKARI-Bench, by its card (the author's own benchmark)
    ('bekko-a25m-256', 'hotchpotch/bekko-embedding-v1-a25m', 256, 'bekko-a25m with its vectors cut to 256 (Matryoshka): a third less index for a small loss (~200 MB).', '', '', 1, null, null, null, null, null),
    ('granite-97m-multilingual', 'ibm-granite/granite-embedding-97m-multilingual-r2', 384, null, '', '', 0, null, null, 0.79, 0.885, 0.82),
    ('granite-small-english', 'onnx-community/granite-embedding-small-english-r2-ONNX', 384, null, '', '', 0, null, null, null, null, null),
    ('granite-english', 'sirasagi62/granite-embedding-english-r2-ONNX', 768, null, '', '', 0, null, null, null, null, null),
    ('e5-base-v2', 'intfloat/e5-base-v2', 768, null, 'query: ', 'passage: ', 0, null, null, null, null, null),
    ('f2llm-160m', 'onnx-community/F2LLM-v2-160M-ONNX', 640, null, 'Instruct: Given a question, retrieve passages that can help answer the question.' || char(10) || 'Query: ', '', 0, null, null, null, null, null),
    -- each Apple GPU profile as its ONNX twin: the same sizes and prefixes
    ('bekko-a25m-mlx', 'hotchpotch/bekko-embedding-v1-a25m:mlx', 384, null, '', '', 0, null, null, null, null, null),
    ('e5-base-v2-mlx', 'intfloat/e5-base-v2:mlx', 768, null, 'query: ', 'passage: ', 0, null, null, null, null, null),
    ('f2llm-160m-mlx', 'codefuse-ai/F2LLM-v2-160M:mlx', 640, null, 'Instruct: Given a question, retrieve passages that can help answer the question.' || char(10) || 'Query: ', '', 0, null, null, null, null, null),
    ('granite-english-gguf', 'mradermacher/granite-embedding-english-r2-GGUF', 768, null, '', '', 0, null, null, null, null, null),
    ('e5-base-v2-gguf', 'ChristianAzinn/e5-base-v2-gguf', 768, null, 'query: ', 'passage: ', 0, null, null, null, null, null),
    ('f2llm-160m-gguf', 'mradermacher/F2LLM-v2-160M-GGUF', 640, null, 'Instruct: Given a question, retrieve passages that can help answer the question.' || char(10) || 'Query: ', '', 0, null, null, null, null, null);

-- Each reranker's floor and score curve (`reranker_calibration`). Uncalibrated until measured:
-- a floor of 0.05 (logit about -3) and the identity curve (beta 1, 1). `mise run
-- calibrate-rerankers` measures both on borderline pairs of your own collections, the way Cohere
-- sets a relevance threshold: the average score of 30 to 50 pairs a person judged borderline
-- relevant.
insert or ignore into reranker_calibration (model, floor, beta_a, beta_b, source)
values
    ('cross-encoder/ms-marco-MiniLM-L2-v2', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('cross-encoder/ettin-reranker-17m-v1', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('cross-encoder/ettin-reranker-32m-v1', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('cross-encoder/ettin-reranker-68m-v1', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('cross-encoder/ettin-reranker-150m-v1', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('Alibaba-NLP/gte-reranker-modernbert-base', 0.05, 1.0, 1.0, 'uncalibrated'),
    ('jrc2139/granite-embedding-reranker-english-r2-ONNX', 0.05, 1.0, 1.0, 'uncalibrated');

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
-- - granite-97m-multilingual: the lowest answered best cosine is 0.793 (haskie docs; the Rust
--   book's is 0.881), so 0.79; it catches 18 of 45 unanswered. The highest unanswered is 0.884 (Rust
--   book), so the band ends at 0.885: it holds the other 27, with 8 of 57 answered (14%). Topics:
--   other pairs <= 0.812, so 0.82, joining 89% of same-topic pairs. Measured on the docs at the
--   pinned commit: a bar holds for the text it was measured on.
-- - Measured against the rerankers' floors above: ettin-32m at 0.05 flags no answered question
--   and catches none of 45 (granite-97m pools). Ettin scores a question its shelf does not answer
--   high when the shelf is near its topic: the Rust book's unanswered questions have a median best
--   score of 0.997 (logit 5.9), its answered ones at least 0.99988 (logit 9.0). On the haskie docs
--   the two overlap (unanswered up to 0.99998). So a reranked search finds no gap until
--   `calibrate-rerankers` sets ettin's floor; the MiniLM-L-6 it replaces caught 36 of 45 at 0.05.
