"""Model lifecycle: one durable download record per model, and what a search may do meanwhile.

Downloaded and usable are two different things (see `models`): the record outlives the process,
the loaded model does not. Every test here takes the `dbos` fixture, because a download is a
workflow, and replaces `models.load_model` rather than `embed.warm` where it can, so no case
waits out the download retry schedule.
"""

import threading
from pathlib import Path

import pytest
from conftest import (
    DESCRIBER,
    MD,
    WAIT,
    attach_document,
    await_terminal,
    collection_hits,
    counted_list_workflows,
    default_model,
    import_document,
    restart_dbos,
    search_with,
    until,
    wait_event,
    wait_for,
)
from dbos import DBOS

from haskie.collection.collection import Collection
from haskie.errors import HaskieError, NotReady, Unavailable
from haskie.indexing import dbos_names, embed, gguf_models, models, operations, workflows
from haskie.indexing.models import ModelKind, ModelLoading
from haskie.settings import (
    Accelerator,
    CollectionOverrides,
    Describer,
    Descriptors,
    Fusion,
    PipelineSettings,
    Reranker,
    SearchMode,
    SearchOverrides,
    SearchSettings,
    UserSettings,
    save_user_settings,
)

pytestmark = pytest.mark.anyio

# OCR is on by default and its model is required then: off wherever a test is about other models
NO_OCR = PipelineSettings(ocr=False)


async def test_no_model_is_required_for_full_text_only(dbos) -> None:
    assert await models.ensure_models(UserSettings(embedding="none", pipeline=NO_OCR)) == []
    assert await models.model_statuses() == [], "and the stored settings ask for none either"


@pytest.mark.parametrize(
    ("name", "outcome", "state", "match", "raised"),
    [
        ("loaded in this process", "ready", "ready", None, None),
        # a failed model stays failed until a restart or a settings save: no retry hint
        (
            "download failed",
            "error",
            "error",
            "failed to load: RuntimeError: no such model",
            Unavailable,
        ),
        ("never required before", "missing", "pending", "is not loaded yet", ModelLoading),
        (
            "still downloading",
            "blocked",
            "loading",
            "is downloading .operation dl:embedding:",
            ModelLoading,
        ),
        (
            "downloaded, caches cold",
            "cold",
            "loading",
            "is loading in this process",
            ModelLoading,
        ),
    ],
)
async def test_model_state_decides_whether_search_may_run(
    dbos,
    monkeypatch,
    name: str,
    outcome: str,
    state: str,
    match: str | None,
    raised: type[Exception] | None,
) -> None:
    """A download record that says SUCCESS is not enough: the model lives in the caches of the
    process that loaded it, so a boot that inherits the record still reports "loading" until it
    has warmed the model itself.

    `load_model` is replaced rather than `embed.warm`, so no case waits out the download retry
    schedule (five attempts, five seconds apart)."""
    blocked = threading.Event()

    async def load_model(kind: str, name_: str) -> None:
        if outcome == "error":
            raise RuntimeError("no such model")
        if outcome == "blocked":
            assert await wait_event(blocked)

    monkeypatch.setattr(models, "load_model", load_model)
    user = await save_user_settings(
        UserSettings(embedding="granite-97m-multilingual", pipeline=NO_OCR)
    )
    model_name = (await default_model()).name

    if outcome != "missing":
        await models.ensure_models(user)
        if outcome != "blocked":
            await await_terminal([models._model_id(ModelKind.EMBEDDING, model_name)])
    if outcome == "cold":
        models._ready.clear()  # the record of a boot whose caches this process does not have

    try:
        (status,) = await models.model_statuses()
        assert (status.kind, status.name, status.state) == ("embedding", model_name, state), name
        if raised is None:
            await models.require_ready(ModelKind.EMBEDDING, model_name)  # no raise
        else:
            with pytest.raises(raised, match=match) as caught:
                await models.require_ready(ModelKind.EMBEDDING, model_name)
            assert type(caught.value) is raised, name
    finally:
        blocked.set()
        await await_terminal([models._model_id(ModelKind.EMBEDDING, model_name)])


async def test_ensure_models_retries_a_model_that_failed(dbos, monkeypatch) -> None:
    attempts: list[str] = []

    async def load_model(kind: str, name: str) -> None:
        attempts.append(name)
        if len(attempts) == 1:
            raise RuntimeError("connection reset")

    monkeypatch.setattr(models, "load_model", load_model)
    user = await save_user_settings(
        UserSettings(embedding="granite-97m-multilingual", pipeline=NO_OCR)
    )
    model_name = (await default_model()).name

    await models.ensure_models(user)
    await await_terminal([models._model_id(ModelKind.EMBEDDING, model_name)])
    assert (await models.model_statuses())[0].state == "error"

    await models.ensure_models(user)  # retries under the same id instead of leaving it failed
    await await_terminal([models._model_id(ModelKind.EMBEDDING, model_name)])

    assert (await models.model_statuses())[0].state == "ready"
    assert attempts == [model_name, model_name], "the retry really called the loader again"
    assert len(await DBOS.list_workflows_async(name=dbos_names.DOWNLOAD_WORKFLOW)) == 1, (
        "same workflow id"
    )


async def test_require_ready_answers_from_the_process_that_loaded_the_model(
    dbos, monkeypatch
) -> None:
    """Every search asks whether the model is ready, and the process that loaded it knows without
    asking the database; a failure is never remembered, because the retry must be visible."""
    attempts: list[str] = []

    async def load_model(kind: str, name: str) -> None:
        attempts.append(name)
        if len(attempts) == 1:
            raise RuntimeError("connection reset")

    monkeypatch.setattr(models, "load_model", load_model)
    user = await save_user_settings(
        UserSettings(embedding="granite-97m-multilingual", pipeline=NO_OCR)
    )
    model_name = (await default_model()).name
    await models.ensure_models(user)
    await await_terminal([models._model_id(ModelKind.EMBEDDING, model_name)])
    with pytest.raises(Unavailable, match="failed to load"):
        await models.require_ready(ModelKind.EMBEDDING, model_name)  # a failure is never cached

    await models.ensure_models(user)  # deletes the failed record and enqueues the same id again
    await await_terminal([models._model_id(ModelKind.EMBEDDING, model_name)])
    calls = counted_list_workflows(monkeypatch)

    await models.require_ready(ModelKind.EMBEDDING, model_name)
    await models.require_ready(ModelKind.EMBEDDING, model_name)
    assert calls == [], "the load ran here, so no search has to look the record up"

    await models.ensure_models(user)  # a healthy record: nothing is deleted or started again
    queries = len(calls)
    await models.require_ready(ModelKind.EMBEDDING, model_name)
    assert len(calls) == queries, "and reapplying the settings does not make the model cold"
    assert attempts == [model_name, model_name], "no third load"


@pytest.mark.network
async def test_embed_stage_precomputes_vectors_and_hybrid_search_uses_them(
    dbos, tmp_path: Path
) -> None:
    user = await save_user_settings(
        UserSettings(
            embedding="granite-97m-multilingual",
            search=SearchSettings(reranker=Reranker.CROSS_ENCODER),
            pipeline=NO_OCR,
        )
    )
    await models.ensure_models(user)
    for kind, name in await models.required(user):
        await wait_for(models._model_id(kind, name))
    assert {(m.kind, m.state) for m in await models.model_statuses()} == {
        ("embedding", "ready"),
        ("reranker", "ready"),
    }
    collection = await Collection.create("vec")
    await collection.set_overrides(CollectionOverrides(chunk_size=40))
    body = "# Cats\n\nCats purr and chase mice.\n\n# Finance\n\nBonds yield interest.\n"
    doc = await import_document(dbos, "v.md", body, tmp_path)
    await attach_document(dbos, "vec", doc.name)

    table = await (await collection.index())._existing()
    assert table is not None and "vector" in (await table.schema()).names
    records = sorted((await table.to_arrow()).to_pylist(), key=lambda r: r["seq"])
    assert [r["headings"] for r in records] == [["Cats"], ["Finance"]]
    assert all(len(r["vector"]) == 384 for r in records)
    assert (await collection_hits("vec", "kitten"))[0].headings[-1] == "Cats", "semantic hit"

    # search options: every mode/fusion answers; fts alone cannot find "kitten"
    semantic = await search_with("vec", "kitten", SearchOverrides(mode=SearchMode.VECTOR))
    assert semantic[0].headings[-1] == "Cats"
    assert await search_with("vec", "kitten", SearchOverrides(mode=SearchMode.FTS)) == []
    linear = await search_with(
        "vec", "bonds", SearchOverrides(mode=SearchMode.HYBRID, fusion=Fusion.LINEAR)
    )
    assert linear[0].headings[-1] == "Finance"
    lexical = await search_with(
        "vec", "bonds", SearchOverrides(fusion=Fusion.LINEAR, vector_weight=0.0, bm25_weight=1.0)
    )
    assert lexical[0].headings[-1] == "Finance"
    (best,) = await search_with("vec", "kitten", SearchOverrides(fusion=Fusion.RRF, limit=1))
    assert best.headings[-1] == "Cats"

    # cross-encoder reranker works on top of any mode, including vector-only and fts
    for mode in (SearchMode.VECTOR, SearchMode.HYBRID):
        reranked = SearchOverrides(mode=mode, reranker=Reranker.CROSS_ENCODER, candidates=10)
        hits = await search_with("vec", "kitten", reranked)
        assert hits[0].headings[-1] == "Cats" and hits[0].score > hits[1].score, mode
        assert all(0 < hit.score < 1 for hit in hits), f"{mode}: the sigmoid of the logit"
    lexical_reranked = await search_with(
        "vec", "bonds", SearchOverrides(mode=SearchMode.FTS, reranker=Reranker.CROSS_ENCODER)
    )
    assert lexical_reranked[0].headings[-1] == "Finance"


@pytest.mark.network
async def test_models_are_idempotent_and_fail_fast_when_missing(dbos, monkeypatch) -> None:
    plain = UserSettings(embedding="none", pipeline=NO_OCR)
    assert await models.ensure_models(plain) == [], "nothing required for full-text only"
    with pytest.raises(NotReady, match="not loaded yet"):
        await models.require_ready(
            ModelKind.EMBEDDING, "ibm-granite/granite-embedding-97m-multilingual-r2"
        )

    wanted = await save_user_settings(
        UserSettings(embedding="granite-97m-multilingual", pipeline=NO_OCR)
    )
    (status,) = await models.ensure_models(wanted)
    assert status.state in ("loading", "ready")
    await wait_for(models._model_id(ModelKind.EMBEDDING, status.name))
    assert (await models.model_statuses())[0].state == "ready"
    before = len(await DBOS.list_workflows_async(name=dbos_names.DOWNLOAD_WORKFLOW))
    await models.ensure_models(wanted)
    assert len(await DBOS.list_workflows_async(name=dbos_names.DOWNLOAD_WORKFLOW)) == before, (
        "no second load"
    )

    # saved past the API, which would refuse a model the catalogue does not hold
    broken = await save_user_settings(
        UserSettings(
            embedding="none",
            search=SearchSettings(reranker=Reranker.CROSS_ENCODER, reranker_model="nope/x"),
            pipeline=NO_OCR,
        )
    )
    await models.ensure_models(broken)
    with pytest.raises(HaskieError):  # the model does not exist
        await wait_for(models._model_id(ModelKind.RERANKER, "nope/x"))
    (status,) = [one for one in await models.model_statuses() if one.name == "nope/x"]
    assert (status.kind, status.state) == ("reranker", "error") and status.error
    with pytest.raises(Unavailable, match="failed to load"):
        await models.require_ready(ModelKind.RERANKER, "nope/x")


@pytest.mark.network
async def test_search_rejects_a_query_while_the_embedding_model_loads(
    dbos, tmp_path: Path, monkeypatch
) -> None:
    """The index has vectors, so a hybrid query needs the model; a request must fail fast with
    503 semantics rather than block on a download."""
    user = await save_user_settings(
        UserSettings(embedding="granite-97m-multilingual", pipeline=NO_OCR)
    )
    await models.ensure_models(user)
    await await_terminal([models._model_id(ModelKind.EMBEDDING, (await default_model()).name)])
    collection = await Collection.create("busy")
    doc = await import_document(dbos, "g.md", MD, tmp_path)
    await attach_document(dbos, "busy", doc.name)
    assert await collection_hits(collection.name, "lancedb"), "ready model answers"

    monkeypatch.setattr(models, "_ready", set())  # as after a restart: caches are cold

    with pytest.raises(NotReady, match="is loading in this process"):
        await collection_hits(collection.name, "lancedb")


@pytest.mark.parametrize(
    ("name", "user", "collection_rerankers", "expected"),
    [
        ("full text only", UserSettings(embedding="none", pipeline=NO_OCR), [], []),
        (
            "an embedding profile",
            UserSettings(embedding="granite-97m-multilingual", pipeline=NO_OCR),
            [],
            [("embedding", "ibm-granite/granite-embedding-97m-multilingual-r2")],
        ),
        (
            "an embedding profile and a cross-encoder",
            UserSettings(
                embedding="granite-97m-multilingual",
                search=SearchSettings(reranker=Reranker.CROSS_ENCODER),
                pipeline=NO_OCR,
            ),
            [],
            [
                ("embedding", "ibm-granite/granite-embedding-97m-multilingual-r2"),
                ("reranker", "cross-encoder/ettin-reranker-32m-v1"),
            ],
        ),
        (
            "a collection override nobody else asks for",
            UserSettings(embedding="none", pipeline=NO_OCR),
            ["Alibaba-NLP/gte-reranker-modernbert-base"],
            [("reranker", "Alibaba-NLP/gte-reranker-modernbert-base")],
        ),
        (
            "the same model at both levels is wanted once",
            UserSettings(
                embedding="none",
                search=SearchSettings(reranker=Reranker.CROSS_ENCODER),
                pipeline=NO_OCR,
            ),
            ["cross-encoder/ettin-reranker-32m-v1", "Alibaba-NLP/gte-reranker-modernbert-base"],
            [
                ("reranker", "cross-encoder/ettin-reranker-32m-v1"),
                ("reranker", "Alibaba-NLP/gte-reranker-modernbert-base"),
            ],
        ),
        (
            "descriptors an llm writes",
            UserSettings(
                embedding="granite-97m-multilingual",
                pipeline=PipelineSettings(ocr=False, descriptors=Descriptors.LLM),
            ),
            [],
            [
                ("embedding", "ibm-granite/granite-embedding-97m-multilingual-r2"),
                ("describer", "ggml-org/gemma-4-E2B-it-GGUF"),
                ("vocabulary", "Qwen/Qwen3-Embedding-0.6B-GGUF"),
            ],
        ),
        (
            "descriptors the other describer writes",
            UserSettings(
                embedding="none",
                pipeline=PipelineSettings(
                    ocr=False, descriptors=Descriptors.LLM, describer=Describer.QWEN_3_5_4B
                ),
            ),
            [],
            [
                ("describer", "unsloth/Qwen3.5-4B-GGUF"),
                ("vocabulary", "Qwen/Qwen3-Embedding-0.6B-GGUF"),
            ],
        ),
        (
            "OCR on, as by default: its model, after the others",
            UserSettings(embedding="granite-97m-multilingual"),
            [],
            [
                ("embedding", "ibm-granite/granite-embedding-97m-multilingual-r2"),
                ("ocr", "pp-ocrv6-small"),
            ],
        ),
    ],
)
async def test_required_models_follow_the_settings(
    dbos,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    user: UserSettings,
    collection_rerankers: list[str],
    expected: list,
) -> None:
    async def overrides(_: UserSettings) -> list[str]:
        return collection_rerankers

    monkeypatch.setattr(models, "_collection_rerankers", overrides)

    assert await models.required(user) == expected, name


async def test_the_describer_is_downloaded_by_its_own_loader(dbos, monkeypatch) -> None:
    """The llm descriptor strategy needs its generator and the vocabulary's embedder: a download
    record of each kind, each loaded by its own loader, after which a run can ask them."""
    loaded: list[str] = []
    monkeypatch.setattr(embed, "warm_generator", lambda name, accelerator: loaded.append(name))
    monkeypatch.setattr(embed, "warm", lambda name, accelerator: loaded.append(f"embed {name}"))
    user = await save_user_settings(
        UserSettings(
            embedding="none", pipeline=PipelineSettings(ocr=False, descriptors=Descriptors.LLM)
        )
    )

    statuses = await models.ensure_models(user)

    assert [(status.kind, status.name) for status in statuses] == [
        ("describer", DESCRIBER),
        ("vocabulary", gguf_models.VOCABULARY_EMBEDDER),
    ]
    wanted = [(ModelKind.DESCRIBER, DESCRIBER)]
    wanted.append((ModelKind.VOCABULARY, gguf_models.VOCABULARY_EMBEDDER))
    await await_terminal([models._model_id(kind, name) for kind, name in wanted])
    assert sorted(loaded) == sorted([DESCRIBER, f"embed {gguf_models.VOCABULARY_EMBEDDER}"])
    for kind, name in wanted:
        await models.require_ready(kind, name)  # no raise


async def test_a_downloaded_model_nothing_requires_warms_when_asked_for(dbos, monkeypatch) -> None:
    """The boot warms only what the settings require now. A run resumed after a change may still
    ask for another downloaded model, the describer after a switch back to c-TF-IDF: asking warms
    it, so the run's wait ends rather than loops."""
    loaded: list[str] = []
    monkeypatch.setattr(embed, "warm_generator", lambda name, accelerator: loaded.append(name))
    monkeypatch.setattr(embed, "warm", lambda name, accelerator: None)  # the vocabulary's
    user = await save_user_settings(
        UserSettings(
            embedding="none", pipeline=PipelineSettings(ocr=False, descriptors=Descriptors.LLM)
        )
    )
    await models.ensure_models(user)
    workflow_id = models._model_id(ModelKind.DESCRIBER, DESCRIBER)
    await await_terminal([workflow_id])
    await save_user_settings(
        UserSettings(embedding="none", pipeline=NO_OCR)
    )  # nothing requires it now
    models._ready.clear()  # a restart: the files stay, the caches do not
    loaded.clear()

    with pytest.raises(NotReady, match="is loading in this process"):
        await models.require_ready(ModelKind.DESCRIBER, DESCRIBER)

    async def warm() -> bool:
        return models.is_warm(workflow_id)

    await until(warm, "the ask never warmed the model")
    assert loaded == [DESCRIBER]
    await models.require_ready(ModelKind.DESCRIBER, DESCRIBER)  # no raise


async def test_collection_reranker_override_is_downloaded(dbos, monkeypatch) -> None:
    """A reranker chosen for one collection is a model the installation needs: nothing else would
    ever fetch it, and the first search of that collection would fail with "not loaded yet"."""
    loaded: list[str] = []
    monkeypatch.setattr(embed, "warm_reranker", lambda name, accelerator: loaded.append(name))
    override = "cross-encoder/ettin-reranker-150m-v1"
    collection = await Collection.create("picky")
    await collection.set_overrides(
        CollectionOverrides(
            search=SearchOverrides(reranker=Reranker.CROSS_ENCODER, reranker_model=override)
        )
    )
    user = await save_user_settings(UserSettings(embedding="none", pipeline=NO_OCR))

    assert await Collection.reranker_overrides(user.search) == [override]
    (status,) = await models.ensure_models(user)

    assert (status.kind, status.name) == ("reranker", override)
    workflow_id = models._model_id(ModelKind.RERANKER, override)
    assert workflow_id.startswith("dl:reranker:")
    await await_terminal([workflow_id])
    assert loaded == [override], "the download workflow really called the loader"
    (download,) = (await operations.list_operations("download")).items
    assert download.id == workflow_id
    assert download.title == f"reranker {override}", "kind and model read out of the id"
    assert (download.status, download.error) == ("SUCCESS", None)
    await models.require_ready(
        ModelKind.RERANKER, override
    )  # no raise: the collection can be searched


async def test_downloads_list_one_row_per_required_model(dbos, monkeypatch) -> None:
    """The list is the read model of the `ensure_model` workflows, so it has one row per model
    the settings ask for, whatever state it is in."""
    monkeypatch.setattr(embed, "warm", lambda name, accelerator: None)
    monkeypatch.setattr(embed, "warm_reranker", lambda name, accelerator: None)
    user = await save_user_settings(
        UserSettings(
            embedding="granite-97m-multilingual",
            search=SearchSettings(reranker=Reranker.CROSS_ENCODER),
            pipeline=NO_OCR,
        )
    )

    await models.ensure_models(user)
    await await_terminal(
        [models._model_id(kind, name) for kind, name in await models.required(user)]
    )

    downloads = (await operations.list_operations("download")).items
    assert {d.title for d in downloads} == {
        f"embedding {(await default_model()).name}",
        "reranker cross-encoder/ettin-reranker-32m-v1",
    }, "one row per required model, kind and name read out of the workflow id"
    assert {d.status for d in downloads} == {"SUCCESS"}
    assert all(d.detail["warm"] for d in downloads), "loaded here, so this process can search"
    assert all(d.created_at > 0 and d.error is None for d in downloads)
    assert [d.created_at for d in downloads] == sorted(
        (d.created_at for d in downloads), reverse=True
    ), "newest first"


async def test_restart_does_not_create_a_second_download_record(dbos, monkeypatch) -> None:
    """The files stay on disk and the record is durable, so a restart reuses both: one row per
    model, however often the dev server reloads."""
    monkeypatch.setattr(embed, "warm", lambda name, accelerator: None)
    user = await save_user_settings(
        UserSettings(embedding="granite-97m-multilingual", pipeline=NO_OCR)
    )
    workflow_id = models._model_id(ModelKind.EMBEDDING, (await default_model()).name)
    await models.ensure_models(user)
    await await_terminal([workflow_id])

    models._ready.clear()  # a new process has the files, not the caches
    await restart_dbos()
    await workflows.apply_settings(user)  # the boot applies them once; twice must change nothing

    (download,) = (await operations.list_operations("download")).items
    assert download.id == workflow_id, "the same record, not one per boot"
    assert download.status == "SUCCESS", "and it is not downloaded again"

    async def warmed() -> bool:
        return (await models.model_statuses())[0].state == "ready"

    await until(warmed, "the model was never warmed")
    assert (await operations.list_operations("download")).items[0].detail["warm"] is True


async def test_a_downloaded_model_is_warmed_after_restart_before_search_uses_it(
    dbos, monkeypatch
) -> None:
    """Warming is a local read of the disk cache, but it still takes seconds, so the boot hands it
    to a background thread. A search that arrives first fails fast with 503 semantics."""
    warming, release = threading.Event(), threading.Event()

    def warm(name, accelerator) -> None:
        # sync, and run in a worker thread through `cpu.on_cpu`: blocking here blocks no loop
        warming.set()
        assert release.wait(timeout=WAIT), "the test never released the warm-up"

    async def load_model(kind, name) -> None:  # the download itself
        return None

    monkeypatch.setattr(models, "load_model", load_model)
    monkeypatch.setattr(embed, "warm", warm)
    user = await save_user_settings(
        UserSettings(embedding="granite-97m-multilingual", pipeline=NO_OCR)
    )
    model_name = (await default_model()).name
    await models.ensure_models(user)
    await await_terminal([models._model_id(ModelKind.EMBEDDING, model_name)])

    models._ready.clear()  # a new process has the files, not the caches
    await restart_dbos()

    assert await wait_event(warming), "the boot warms the model it already downloaded"
    with pytest.raises(NotReady, match="is loading in this process"):
        await models.require_ready(ModelKind.EMBEDDING, model_name)
    assert (await models.model_statuses())[0].state == "loading"

    release.set()

    async def loaded() -> bool:
        return models.is_warm(models._model_id(ModelKind.EMBEDDING, model_name))

    await until(loaded, "never warmed")
    await models.require_ready(ModelKind.EMBEDDING, model_name)  # no raise: the search may run now


async def test_a_knowledge_model_waits_for_its_first_use_and_is_freed_once_idle(
    dbos, monkeypatch
) -> None:
    """Downloaded at boot, a knowledge model is not loaded: `downloaded`, in its group. The first
    run that asks loads it (`ready`); once nobody used it for `IDLE_SECONDS`, `free_idle` frees
    it and it is `downloaded` again, and the next ask loads it again. A search model is warmed at
    boot and never freed."""
    monkeypatch.setattr(gguf_models, "available", lambda: True)  # llama.cpp stood in for
    built: list[str] = []

    class Model:
        def __init__(self, name: str) -> None:
            built.append(name)
            self.closed = False

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(gguf_models, "GgufGenerator", Model)
    monkeypatch.setattr(gguf_models, "GgufEmbedder", Model)
    monkeypatch.setattr(embed, "_on_demand", embed.OnDemand())
    monkeypatch.setattr(embed, "warm", lambda name, accelerator: None)  # the search model
    user = await save_user_settings(
        UserSettings(
            embedding="granite-97m-multilingual",
            pipeline=PipelineSettings(ocr=False, descriptors=Descriptors.LLM),
        )
    )
    monkeypatch.setattr(models, "load_model", _noop_load)
    await models.ensure_models(user)
    ids = [models._model_id(kind, name) for kind, name in await models.required(user)]
    await await_terminal(ids)
    models._ready.clear()  # a restart: records SUCCESS, every cache cold
    await models.ensure_models(user)  # the boot: warms the search model alone
    describer = models._model_id(ModelKind.DESCRIBER, DESCRIBER)

    async def states() -> dict[str, tuple[str, str]]:
        return {one.name: (one.group, one.state) for one in await models.model_statuses()}

    async def warm() -> bool:
        return (await states())[(await default_model()).name][1] == "ready"

    await until(warm, "the search model never warmed")
    assert (await states())[DESCRIBER] == ("knowledge", "downloaded")
    assert (await states())[gguf_models.VOCABULARY_EMBEDDER] == ("knowledge", "downloaded")
    assert (await states())[(await default_model()).name][0] == "search"
    assert built == [], "no knowledge model loaded at boot"

    with pytest.raises(ModelLoading, match="loading in this process"):
        await models.require_ready(ModelKind.DESCRIBER, DESCRIBER)
    await until(lambda: _async(models.is_warm(describer)), "the ask never loaded it")
    assert built == [DESCRIBER]
    assert (await states())[DESCRIBER][1] == "ready"

    assert models.free_idle() == [], "used a moment ago: kept"
    monkeypatch.setattr(models, "IDLE_SECONDS", 0)
    assert models.free_idle() == [DESCRIBER]
    assert (await states())[DESCRIBER][1] == "downloaded"
    assert not models.is_warm(describer), "forgotten, so the next ask loads it again"
    with pytest.raises(ModelLoading):
        await models.require_ready(ModelKind.DESCRIBER, DESCRIBER)
    await until(lambda: _async(models.is_warm(describer)), "the second ask never loaded it")
    assert built == [DESCRIBER] * 2


async def _noop_load(kind: ModelKind, name: str) -> None:
    """The download, stood in for: its record says SUCCESS, nothing is loaded."""


async def _async(value: bool) -> bool:
    return value


def test_a_knowledge_model_in_use_is_never_freed(monkeypatch) -> None:
    """A describe batch holds its model across many prompts: `free_idle` skips one in use however
    long ago it was taken, frees it once released and idle, and closes what it frees."""
    monkeypatch.setattr(gguf_models, "available", lambda: True)
    closed: list[str] = []

    class Model:
        def __init__(self, name: str) -> None:
            assert not embed._MODEL_LOCK.locked(), "a search would wait out the load"
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    on_demand = embed.OnDemand()
    key = (DESCRIBER, Accelerator.AUTO)
    with on_demand.use(key, lambda: Model(DESCRIBER)) as model:
        assert on_demand.loaded(DESCRIBER)
        assert on_demand.free_idle(0) == [], "in use"
        with on_demand.use(key, lambda: pytest.fail("built twice")) as again:
            assert again is model, "the loaded one, shared"
    assert on_demand.free_idle(3600) == [], "released a moment ago"
    assert on_demand.free_idle(0) == [DESCRIBER]
    assert closed == [DESCRIBER] and not on_demand.loaded(DESCRIBER)
    assert on_demand.free_idle(0) == [], "nothing left to free"


async def test_the_ocr_model_is_usable_once_downloaded_and_after_a_restart(
    dbos, monkeypatch
) -> None:
    """OCR has a group of its own: downloaded with the others, and usable as soon as it is on
    disk, since each conversion loads it in its own worker. A restart finds it usable without
    loading anything, and the status says so: downloaded, on the CPU."""
    from haskie.document import ocr
    from haskie.indexing.hardware import Device

    fetched: list[str] = []
    monkeypatch.setattr(ocr, "fetch", lambda name, _: fetched.append(name))
    user = await save_user_settings(UserSettings(embedding="none"))
    workflow_id = models._model_id(ModelKind.OCR, ocr.MODEL)

    await models.ensure_models(user)
    await await_terminal([workflow_id])
    models._ready.clear()  # a new process
    (status,) = await models.ensure_models(user)  # its boot

    assert (status.kind, status.group, status.state, status.device) == (
        "ocr",
        "conversion",
        "downloaded",  # never in this process's memory: each conversion loads it
        Device.CPU,
    )
    assert fetched == [ocr.MODEL], "downloaded once, and nothing loaded at the restart"
    assert not models.is_warm(workflow_id), "nor marked as loaded in this process"
    await models.require_ready(ModelKind.OCR, ocr.MODEL)  # no raise
