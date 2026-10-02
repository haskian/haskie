"""The model catalogue, read out of a real migrated home: what the seed holds, the order the
pickers list it in, how a profile resolves, and what a write naming an unknown model hits.
"""

import sqlite3
from datetime import date
from pathlib import Path

import msgspec
import pytest
from sqlalchemy.exc import IntegrityError

from haskie import db
from haskie.catalogue import catalogue
from haskie.catalogue.catalogue import DuplicateCosine, EmbeddingModel
from haskie.errors import InvalidInput
from haskie.indexing import gguf_models, mlx_models, onnx_models
from haskie.indexing.hardware import Device, Runtime
from haskie.settings import (
    DEFAULT_RERANKER,
    Accelerator,
    CollectionOverrides,
    Descriptors,
    PipelineSettings,
    Reranker,
    SearchOverrides,
    SearchSettings,
    UserSettings,
)

pytestmark = pytest.mark.anyio

ON_CPU = PipelineSettings(accelerator=Accelerator.CPU)

BEKKO = "hotchpotch/bekko-embedding-v1-a25m"


async def test_every_model_says_what_it_is() -> None:
    """An embedder or reranker is chosen by what its metadata says, so none may go without it."""
    embedders = await catalogue.embedding_metadata()
    rerankers = await catalogue.rerankers()

    assert (len(embedders), len(rerankers)) == (14, 7), "every profile and reranker of the seed"
    assert all(isinstance(one, catalogue.EmbedderMetadata) for one in embedders.values())
    assert all(isinstance(one, catalogue.RerankerMetadata) for one in rerankers.values())
    for name, metadata in [*embedders.items(), *rerankers.items()]:
        assert metadata.parameters > 0 and metadata.context_tokens > 0, name
        assert all((metadata.description, metadata.languages, metadata.license)), name
        assert metadata.model_card_url.startswith("https://huggingface.co/"), name
        assert metadata.devices, f"{name}: runs somewhere"
    assert embedders["granite-97m-multilingual"] == catalogue.EmbedderMetadata(
        description=(
            "The best all-round small multilingual embedder: #1 on multilingual and reasoning "
            "retrieval; a good default (~390 MB)."
        ),
        parameters=97441152,
        context_tokens=32768,
        languages="multilingual (200+, 52 enhanced)",
        license="Apache-2.0",
        released=date(2026, 4, 20),
        model_card_url="https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2",
        runtime=Runtime.ONNX,
        devices=(Device.CPU, Device.APPLE_SILICON, Device.GPU),
        dimensions=384,
    )
    # a conversion links to and is dated by the weights it was made from
    granite = "jrc2139/granite-embedding-reranker-english-r2-ONNX"
    assert rerankers[granite] == catalogue.RerankerMetadata(
        description=(
            "IBM's reranker, trained without MS MARCO and its non-commercial license (~600 MB)."
        ),
        parameters=149605633,
        context_tokens=8192,
        languages="English",
        license="Apache-2.0",
        released=date(2025, 8, 4),
        model_card_url="https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2",
        runtime=Runtime.ONNX,
        devices=(Device.CPU, Device.APPLE_SILICON, Device.GPU),
    )


async def test_models_are_listed_smallest_first() -> None:
    """Every picker lists them in this order: embedders by vector size and, at one size, by
    parameters; rerankers by parameters, MiniLM-L2 the smallest."""
    embedders = await catalogue.embedders()
    metadata = await catalogue.embedding_metadata()
    rerankers = await catalogue.rerankers()

    assert list(metadata) == list(embedders), "metadata and models in one order"
    assert all(metadata[p].dimensions == embedders[p].dims for p in embedders), "one size"
    ranked = [(metadata[p].dimensions, metadata[p].parameters) for p in embedders]
    assert ranked == sorted(ranked)
    assert [one.parameters for one in rerankers.values()] == sorted(
        one.parameters for one in rerankers.values()
    )
    assert list(rerankers)[:3] == [
        "cross-encoder/ms-marco-MiniLM-L2-v2",
        "cross-encoder/ettin-reranker-17m-v1",
        DEFAULT_RERANKER,
    ], "MiniLM-L2 is the smallest, the default the third, a little slower than ettin-17m"


async def test_every_model_has_a_loader_and_every_pin_a_model() -> None:
    """The catalogue lives in the database and the loaders in code, so this keeps the two in step:
    a row nothing can load would fail its first download, and a pin without a row is dead code."""
    embedders = {model.name for model in (await catalogue.embedders()).values()}
    rerankers = set(await catalogue.rerankers())
    pinned_embedders = (
        set(onnx_models.EMBEDDERS)
        | set(mlx_models.POOLED)
        | set(mlx_models.JINA_V5)
        | set(gguf_models.PINS)
    )
    pinned_rerankers = set(onnx_models.RERANKERS) | set(mlx_models.RERANKERS)

    assert (embedders, rerankers) == (pinned_embedders, pinned_rerankers)
    # a generator is no catalogue model, and no embedder pin either: one name, one runtime role
    assert set(gguf_models.GENERATORS).isdisjoint(embedders | rerankers | pinned_embedders)
    assert {gguf_models.describer(one) for one in Descriptors} - {None} == set(
        gguf_models.GENERATORS
    ), "every generator is some strategy's describer"
    # each loader cuts a text at the model's own context (a GGUF file holds no positions past
    # it, a BERT none past 512), which the catalogue also states: the two must agree
    metadata = await catalogue.embedding_metadata()
    contexts = {
        model.name: metadata[profile].context_tokens
        for profile, model in (await catalogue.embedders()).items()
    }
    pins = onnx_models.EMBEDDERS | mlx_models.POOLED | gguf_models.PINS
    assert {name: pin.tokens for name, pin in pins.items()} == {
        name: contexts[name] for name in pins
    }


async def test_the_seed_holds_to_its_own_references() -> None:
    """The seed runs on a connection without foreign keys switched on, so this is what checks
    that every profile names a model that is there, and an embedder at that."""
    async with db.connect() as conn:
        broken = (await conn.exec_driver_sql("pragma foreign_key_check")).all()
        kinds = (
            await conn.exec_driver_sql(
                "select distinct m.kind from embedding_profiles p join models m on m.name = p.model"
            )
        ).all()

    assert list(broken) == [] and list(kinds) == [("embedder",)]


async def test_every_reranker_has_its_calibration_and_only_a_reranker() -> None:
    """The floor a search drops chunks under comes from here when the settings leave it unset, so
    a reranker without a row would silently fall back to the defaults."""
    rerankers = await catalogue.rerankers()
    async with db.connect() as conn:
        rows = await conn.exec_driver_sql(
            "select c.model, m.kind from reranker_calibration c join models m on m.name = c.model"
        )
        calibrated = {model: kind for model, kind in rows}

    assert set(calibrated) == set(rerankers), "one row for every reranker"
    assert set(calibrated.values()) == {"reranker"}, "and for no embedder"
    for name in rerankers:
        assert await catalogue.calibration(name) == catalogue.UNCALIBRATED, name
    assert await catalogue.calibration("no/such-model") == catalogue.UNCALIBRATED, "a default"


def test_the_seed_replays_harmlessly(tmp_path: Path) -> None:
    """`migrate` stamps the version last, so a crash partway leaves 0 and the next boot runs the
    schema and the seed again over what the first run wrote."""
    conn = sqlite3.connect(tmp_path / "replay.db")
    try:
        for _ in range(2):
            conn.executescript(db.schema_ddl())
            conn.executescript(db.SEED.read_text(encoding="utf-8"))
        counts = [
            conn.execute(f"select count(*) from {t}").fetchone()[0]
            for t in ("models", "embedding_profiles")
        ]
    finally:
        conn.close()

    assert counts == [20, 14], "13 embedders and 7 rerankers, 14 profiles: once each"


_MODEL = (
    "insert into models (name, kind, description, parameters, context_tokens, languages, license, "
    "released, model_card_url) values ('{name}', '{kind}', 'd', {parameters}, {context_tokens}, "
    "'English', 'MIT', '{released}', '{url}')"
)
_VALID = {
    "name": "x/y",
    "kind": "reranker",
    "parameters": 1,
    "context_tokens": 512,
    "released": "2024-03-15",
    "url": "https://huggingface.co/x/y",
}


async def test_the_schema_takes_a_valid_model() -> None:
    """The base the refusals below change one field of, so each refusal is that field's."""
    async with db.connect() as conn:
        await conn.exec_driver_sql(_MODEL.format_map(_VALID))


@pytest.mark.parametrize(
    ("name", "sql"),
    [
        (
            "one duplicate cosine without the other",
            "insert into embedding_profiles (profile, model, dims, duplicate_chunk) "
            f"values ('half', '{BEKKO}', 384, 0.9)",
        ),
        (
            "full-text only is the absence of a model, not a row",
            "insert into embedding_profiles (profile, model, dims) "
            f"values ('none', '{BEKKO}', 384)",
        ),
        (
            "a model that is neither kind",
            _MODEL.format_map(_VALID | {"kind": "tokenizer"}),
        ),
        ("a model with no parameters", _MODEL.format_map(_VALID | {"parameters": 0})),
        ("a model that reads no tokens", _MODEL.format_map(_VALID | {"context_tokens": 0})),
        ("a release date not in ISO form", _MODEL.format_map(_VALID | {"released": "15.3.2024"})),
        ("a release date that is no day", _MODEL.format_map(_VALID | {"released": "2024-02-30"})),
        ("a card off the Hub", _MODEL.format_map(_VALID | {"url": "https://example.com/x/y"})),
        (
            "a borderline band that ends under its start",
            "update embedding_profiles set weak_match = 0.7, answered_match = 0.6 "
            "where profile = 'granite-97m-multilingual'",
        ),
    ],
)
async def test_the_schema_refuses_a_row_the_catalogue_cannot_mean(name: str, sql: str) -> None:
    with pytest.raises(IntegrityError):
        async with db.connect() as conn:
            await conn.exec_driver_sql(sql)


@pytest.mark.parametrize(
    ("name", "settings", "expected"),
    [
        ("full-text only has no model", UserSettings(), None),
        (
            "a profile with thresholds, on the hardware the settings choose",
            UserSettings(embedding="granite-97m-multilingual", pipeline=ON_CPU),
            EmbeddingModel(
                "ibm-granite/granite-embedding-97m-multilingual-r2",
                384,
                accelerator=Accelerator.CPU,
                profile="granite-97m-multilingual",
                weak_match=0.79,
                answered_match=0.885,
                same_topic=0.82,
            ),
        ),
        (
            "a model with prefixes",
            UserSettings(embedding="e5-base-v2"),
            EmbeddingModel(
                "intfloat/e5-base-v2",
                768,
                query_prefix="query: ",
                document_prefix="passage: ",
                profile="e5-base-v2",
            ),
        ),
        (
            "a model cut to its first values (Matryoshka)",
            UserSettings(embedding="bekko-a25m-256"),
            EmbeddingModel(BEKKO, 256, matryoshka=True, profile="bekko-a25m-256"),
        ),
        (
            "the same model whole: no cut",
            UserSettings(embedding="bekko-a25m"),
            EmbeddingModel(BEKKO, 384, profile="bekko-a25m"),
        ),
    ],
)
async def test_a_profile_resolves_to_its_model(
    name: str, settings: UserSettings, expected: EmbeddingModel | None
) -> None:
    assert await catalogue.embedding_model(settings) == expected, name


async def test_a_profile_the_catalogue_lacks_does_not_resolve() -> None:
    with pytest.raises(InvalidInput, match="unknown embedding profile: gone"):
        await catalogue.embedding_model(UserSettings(embedding="gone"))


async def test_one_model_cut_two_ways_is_two_profiles() -> None:
    metadata = await catalogue.embedding_metadata()

    whole, cut = metadata["bekko-a25m"], metadata["bekko-a25m-256"]
    assert cut.description.startswith("bekko-a25m with its vectors cut to 256")
    assert whole.description.startswith("Small and multilingual: #1 on RTEB")
    assert (whole.dimensions, cut.dimensions) == (384, 256)
    unshared = {"description": "", "dimensions": 0}
    assert msgspec.structs.replace(cut, **unshared) == msgspec.structs.replace(whole, **unshared), (
        "one model's facts"
    )


# The catalogue holds no MLX reranker today, but the runtime stays: this one stands in
MLX_RERANKER = "test/tiny-reranker-mlx"
SHORT_RERANKER = "test/short-reranker"  # an ONNX reranker that reads less than its loader cuts to


@pytest.fixture
async def stand_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    """An MLX reranker and a short ONNX one, in the catalogue and pinned in code, with MLX and
    llama.cpp installed."""
    async with db.connect() as conn:
        for name, tokens in [(MLX_RERANKER, 8192), (SHORT_RERANKER, 256)]:
            values = _VALID | {"name": name, "kind": "reranker", "context_tokens": tokens}
            await conn.exec_driver_sql(_MODEL.format_map(values))
    monkeypatch.setitem(mlx_models.RERANKERS, MLX_RERANKER, "0")
    monkeypatch.setitem(onnx_models.RERANKERS, SHORT_RERANKER, onnx_models.RerankerPin("0"))
    monkeypatch.setattr(gguf_models, "available", lambda: True)
    monkeypatch.setattr(mlx_models, "available", lambda: True)


@pytest.mark.parametrize(
    ("name", "settings", "error"),
    [
        (
            "an ONNX profile on the CPU runs",
            UserSettings(embedding="granite-97m-multilingual", pipeline=ON_CPU),
            None,
        ),
        (
            "a GGUF profile on the CPU: nowhere to run",
            UserSettings(embedding="e5-base-v2-gguf", pipeline=ON_CPU),
            "ChristianAzinn/e5-base-v2-gguf runs on gguf on the Apple GPU",
        ),
        ("a GGUF profile where llama.cpp runs", UserSettings(embedding="e5-base-v2-gguf"), None),
        (
            "an MLX profile on the CPU: nowhere to run",
            UserSettings(embedding="bekko-a25m-mlx", pipeline=ON_CPU),
            "hotchpotch/bekko-embedding-v1-a25m:mlx runs on mlx on the Apple GPU",
        ),
        (
            "an MLX reranker on the CPU: nowhere to run",
            UserSettings(
                search=SearchSettings(reranker=Reranker.CROSS_ENCODER, reranker_model=MLX_RERANKER),
                pipeline=ON_CPU,
            ),
            f"{MLX_RERANKER} runs on mlx on the Apple GPU",
        ),
        (
            "the same reranker switched off: nothing uses it",
            UserSettings(search=SearchSettings(reranker_model=MLX_RERANKER), pipeline=ON_CPU),
            None,
        ),
        (
            "descriptors an llm writes, on the CPU: its describer has nowhere to run",
            UserSettings(
                pipeline=PipelineSettings(accelerator=Accelerator.CPU, descriptors=Descriptors.LLM)
            ),
            "ggml-org/gemma-4-E2B-it-GGUF runs on gguf on the Apple GPU",
        ),
        (
            "descriptors an llm writes, where llama.cpp runs",
            UserSettings(pipeline=PipelineSettings(descriptors=Descriptors.LLM)),
            None,
        ),
        (
            "a collection's reranker is not checked against the hardware",
            CollectionOverrides(search=SearchOverrides(reranker_model=MLX_RERANKER)),
            None,
        ),
    ],
)
@pytest.mark.usefixtures("stand_ins")
async def test_check_refuses_settings_that_leave_a_model_nowhere_to_run(
    name: str, settings: UserSettings | CollectionOverrides, error: str | None
) -> None:
    if error is None:
        assert await catalogue.check(settings) is None, name
    else:
        with pytest.raises(InvalidInput, match=error):
            await catalogue.check(settings)


@pytest.mark.parametrize(
    ("name", "settings", "error"),
    [
        ("the defaults", UserSettings(), None),
        (
            "a known profile and reranker",
            UserSettings(
                embedding="granite-english",
                search=SearchSettings(reranker_model="Alibaba-NLP/gte-reranker-modernbert-base"),
            ),
            None,
        ),
        ("an unknown profile", UserSettings(embedding="gone"), "unknown embedding profile: gone"),
        (
            "an unknown reranker",
            UserSettings(search=SearchSettings(reranker_model="no/such")),
            "unknown reranker model: no/such",
        ),
        (
            "an embedder named as a reranker",
            UserSettings(
                search=SearchSettings(
                    reranker_model="ibm-granite/granite-embedding-97m-multilingual-r2"
                )
            ),
            "unknown reranker model: ibm-granite/granite-embedding-97m-multilingual-r2",
        ),
        (
            "both unknown: one message names both",
            UserSettings(embedding="gone", search=SearchSettings(reranker_model="no/such")),
            "unknown embedding profile: gone; unknown reranker model: no/such",
        ),
        ("a collection that overrides no model", CollectionOverrides(), None),
        (
            "a collection overriding a known reranker",
            CollectionOverrides(search=SearchOverrides(reranker_model=DEFAULT_RERANKER)),
            None,
        ),
        (
            "a collection overriding an unknown reranker",
            CollectionOverrides(search=SearchOverrides(reranker_model="no/such")),
            "unknown reranker model: no/such",
        ),
    ],
)
async def test_check_names_the_model_the_catalogue_lacks(
    name: str, settings: UserSettings | CollectionOverrides, error: str | None
) -> None:
    if error is None:
        assert await catalogue.check(settings) is None, name
    else:
        with pytest.raises(InvalidInput, match=error):
            await catalogue.check(settings)


# --- the cache name ---------------------------------------------------------------

BASE = EmbeddingModel(
    "intfloat/e5-base-v2", 768, query_prefix="query: ", document_prefix="passage: "
)


@pytest.mark.parametrize(
    ("name", "change", "moves"),
    [
        ("nothing changed", {}, False),
        ("another document prefix", {"document_prefix": "document: "}, True),
        ("another vector size", {"dims": 512}, True),
        ("a Matryoshka cut", {"matryoshka": True}, True),
        ("another query prefix: queries only", {"query_prefix": "search_query: "}, False),
        ("other hardware", {"accelerator": Accelerator.CPU}, False),
        ("other thresholds", {"duplicate": DuplicateCosine(0.9, 0.9)}, False),
    ],
)
def test_the_cache_name_moves_with_what_shapes_a_stored_vector(
    name: str, change: dict, moves: bool
) -> None:
    """The cache is keyed by it, so whatever changes a stored vector must change it, and nothing
    else may: a moved name recomputes every cached embedding of the model."""
    changed = msgspec.structs.replace(BASE, **change)

    assert (changed.cache_name != BASE.cache_name) is moves, name
    assert changed.cache_name.startswith(f"{BASE.name}@{changed.dims}:"), (
        "readable where it is shown"
    )


def test_a_cut_is_keyed_apart_from_the_same_size_whole() -> None:
    """Two models at one size differ in their vectors when one is a cut: the hash tells them
    apart where the readable part cannot."""
    whole = EmbeddingModel("test/tiny", 2)
    cut = EmbeddingModel("test/tiny", 2, matryoshka=True)

    assert whole.cache_name != cut.cache_name


@pytest.mark.parametrize(
    ("name", "model", "reads"),
    [
        ("what the catalogue says a reranker takes, under its cut", SHORT_RERANKER, 256),
        (
            "an ONNX reranker reads what haskie cuts it to",
            "Alibaba-NLP/gte-reranker-modernbert-base",
            512,
        ),
        ("a headed ONNX reranker the same", DEFAULT_RERANKER, 512),
        ("an MLX reranker reads what its loader cuts to", MLX_RERANKER, 512),
    ],
)
@pytest.mark.usefixtures("stand_ins")
async def test_a_whole_excerpt_is_judged_against_what_its_reranker_reads(
    name: str, model: str, reads: int
) -> None:
    from haskie.search import retrieval

    assert await retrieval._reads(model) == reads, name
