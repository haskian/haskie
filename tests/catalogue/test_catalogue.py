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
from haskie.catalogue.catalogue import DuplicateCosine, EmbeddingModel, Matryoshka
from haskie.errors import InvalidInput
from haskie.indexing import embed, gguf_models, mlx_models, onnx_rerank
from haskie.indexing.hardware import Device, Runtime
from haskie.settings import (
    DEFAULT_RERANKER,
    Accelerator,
    CollectionOverrides,
    PipelineSettings,
    Reranker,
    SearchOverrides,
    SearchSettings,
    UserSettings,
)

pytestmark = pytest.mark.anyio

NOMIC = "nomic-ai/nomic-embed-text-v1.5"


async def test_every_model_says_what_it_is() -> None:
    """An embedder or reranker is chosen by what its metadata says, so none may go without it."""
    embedders = await catalogue.embedding_metadata()
    rerankers = await catalogue.rerankers()

    assert (len(embedders), len(rerankers)) == (18, 11), "every profile and reranker of the seed"
    assert all(isinstance(one, catalogue.EmbedderMetadata) for one in embedders.values())
    assert all(isinstance(one, catalogue.RerankerMetadata) for one in rerankers.values())
    for name, metadata in [*embedders.items(), *rerankers.items()]:
        assert metadata.parameters > 0 and metadata.context_tokens > 0, name
        assert all((metadata.description, metadata.languages, metadata.license)), name
        assert metadata.model_card_url.startswith("https://huggingface.co/"), name
        assert metadata.devices, f"{name}: runs somewhere"
    assert embedders["compact"] == catalogue.EmbedderMetadata(
        description="Small and fast; a good default (~130 MB).",
        parameters=33360512,
        context_tokens=512,
        languages="English",
        license="MIT",
        released=date(2023, 9, 12),
        model_card_url="https://huggingface.co/BAAI/bge-small-en-v1.5",
        runtime=Runtime.ONNX,
        devices=(Device.CPU, Device.APPLE_SILICON, Device.GPU),
        dimensions=384,
    )
    # a conversion links to and is dated by the weights it was made from
    assert rerankers["soichisumi/bge-reranker-v2-m3-mlx-affine8"] == catalogue.RerankerMetadata(
        description="BAAI's multilingual reranker, 8-bit (near lossless, 607 MB).",
        parameters=567755777,
        context_tokens=8192,
        languages="multilingual (100+)",
        license="Apache-2.0",
        released=date(2024, 3, 15),
        model_card_url="https://huggingface.co/BAAI/bge-reranker-v2-m3",
        runtime=Runtime.MLX,
        devices=(Device.APPLE_SILICON,),
    )


async def test_models_are_listed_smallest_first() -> None:
    """Every picker lists them in this order: embedders by vector size and, at one size, by
    parameters; rerankers by parameters, the smallest the default."""
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
    assert next(iter(rerankers)) == DEFAULT_RERANKER, "the default stays the smallest"


async def test_every_model_has_a_loader_and_every_pin_a_model() -> None:
    """The catalogue lives in the database and the loaders in code, so this keeps the two in step:
    a row nothing can load would fail its first download, and a pin without a row is dead code."""
    from fastembed import TextEmbedding
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    embedders = {model.name for model in (await catalogue.embedders()).values()}
    rerankers = set(await catalogue.rerankers())
    listed_embedders = {one["model"] for one in TextEmbedding.list_supported_models()}
    listed_rerankers = {one["model"] for one in TextCrossEncoder.list_supported_models()}
    pinned_embedders = (
        set(embed.CUSTOM_EMBEDDERS) | set(mlx_models.EMBEDDERS) | set(gguf_models.PINS)
    )
    pinned_rerankers = (
        set(embed.CUSTOM_RERANKERS) | set(onnx_rerank.REVISIONS) | set(mlx_models.RERANKERS)
    )

    assert embedders - listed_embedders - pinned_embedders == set()
    assert rerankers - listed_rerankers - pinned_rerankers == set()
    assert pinned_embedders <= embedders and pinned_rerankers <= rerankers
    # a GGUF file holds no positions past its model's context, which the catalogue also states
    metadata = await catalogue.embedding_metadata()
    contexts = {
        model.name: metadata[profile].context_tokens
        for profile, model in (await catalogue.embedders()).items()
    }
    assert {name: pin.tokens for name, pin in gguf_models.PINS.items()} == {
        name: contexts[name] for name in gguf_models.PINS
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

    assert counts == [28, 18], "17 embedders and 11 rerankers, 18 profiles: once each"


_MODEL = (
    "insert into models (name, kind, description, parameters, context_tokens, languages, license, "
    "released, model_card_url) values ('x/y', '{kind}', 'd', {parameters}, {context_tokens}, "
    "'English', 'MIT', '{released}', '{url}')"
)
_VALID = {
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
            f"values ('half', '{NOMIC}', 768, 0.9)",
        ),
        (
            "full-text only is the absence of a model, not a row",
            "insert into embedding_profiles (profile, model, dims) "
            f"values ('none', '{NOMIC}', 768)",
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
            UserSettings(
                embedding="compact", pipeline=PipelineSettings(accelerator=Accelerator.CPU)
            ),
            EmbeddingModel(
                "BAAI/bge-small-en-v1.5",
                384,
                accelerator=Accelerator.CPU,
                duplicate=DuplicateCosine(chunk=0.92, passage=0.95),
            ),
        ),
        (
            "a model with prefixes, cut with nomic's layer norm",
            UserSettings(embedding="nomic-v1.5-512"),
            EmbeddingModel(
                NOMIC,
                512,
                query_prefix="search_query: ",
                document_prefix="search_document: ",
                matryoshka=Matryoshka(layer_norm=True),
            ),
        ),
        (
            "the same model whole: no cut, no thresholds",
            UserSettings(embedding="nomic-v1.5"),
            EmbeddingModel(
                NOMIC, 768, query_prefix="search_query: ", document_prefix="search_document: "
            ),
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

    whole, cut = metadata["nomic-v1.5"], metadata["nomic-v1.5-512"]
    assert cut.description.startswith("nomic v1.5 with its vectors cut to 512")
    assert whole.description == "Long passages, open training data (~520 MB)."
    assert (whole.dimensions, cut.dimensions) == (768, 512)
    unshared = {"description": "", "dimensions": 0}
    assert msgspec.structs.replace(cut, **unshared) == msgspec.structs.replace(whole, **unshared), (
        "one model's facts"
    )


MLX_RERANKER = "soichisumi/bge-reranker-v2-m3-mlx-affine8"
ON_CPU = PipelineSettings(accelerator=Accelerator.CPU)


@pytest.mark.parametrize(
    ("name", "settings", "error"),
    [
        (
            "an ONNX profile on the CPU runs",
            UserSettings(embedding="compact", pipeline=ON_CPU),
            None,
        ),
        (
            "a GGUF profile on the CPU: nowhere to run",
            UserSettings(embedding="bge-small-gguf", pipeline=ON_CPU),
            "ggml-org/bge-small-en-v1.5-Q8_0-GGUF runs on gguf on the Apple GPU",
        ),
        (
            "a GGUF profile where llama.cpp runs",
            UserSettings(embedding="bge-small-gguf"),
            None,
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
            "a collection's reranker is not checked against the hardware",
            CollectionOverrides(search=SearchOverrides(reranker_model=MLX_RERANKER)),
            None,
        ),
    ],
)
async def test_check_refuses_settings_that_leave_a_model_nowhere_to_run(
    name: str, settings: UserSettings | CollectionOverrides, error: str | None, monkeypatch
) -> None:
    monkeypatch.setattr(gguf_models, "available", lambda: True)
    monkeypatch.setattr(mlx_models, "available", lambda: True)

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
                embedding="quality", search=SearchSettings(reranker_model="BAAI/bge-reranker-base")
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
            UserSettings(search=SearchSettings(reranker_model="BAAI/bge-small-en-v1.5")),
            "unknown reranker model: BAAI/bge-small-en-v1.5",
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
    NOMIC, 768, query_prefix="search_query: ", document_prefix="search_document: "
)


@pytest.mark.parametrize(
    ("name", "change", "moves"),
    [
        ("nothing changed", {}, False),
        ("another document prefix", {"document_prefix": "passage: "}, True),
        ("another vector size", {"dims": 512}, True),
        ("a Matryoshka cut", {"matryoshka": Matryoshka()}, True),
        ("the cut with a layer norm", {"matryoshka": Matryoshka(layer_norm=True)}, True),
        ("another query prefix: queries only", {"query_prefix": "query: "}, False),
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
    assert changed.cache_name.startswith(f"{NOMIC}@{changed.dims}:"), "readable where it is shown"


def test_a_cut_is_keyed_apart_from_the_same_size_whole() -> None:
    """Two models at one size differ in their vectors when one is a cut: the hash tells them
    apart where the readable part cannot."""
    whole = EmbeddingModel("test/tiny", 2)
    cut = EmbeddingModel("test/tiny", 2, matryoshka=Matryoshka())

    assert whole.cache_name != cut.cache_name
