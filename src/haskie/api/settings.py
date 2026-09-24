"""Init, user settings and the option catalogue the UI renders its forms from."""

import msgspec
from litestar import get, post, put

from haskie import audit, home
from haskie.collection.collection import ACTIVE_MEMBER_STATUSES, MEMBER_STATUSES, MemberStatus
from haskie.document.document import ACTIVE_DOCUMENT_STATUSES, DOCUMENT_STATUSES, DocumentStatus
from haskie.errors import Conflict
from haskie.indexing import mlx_models, models, operations, workflows
from haskie.indexing.dbos_names import ACTIVE_STATUS, RunStatus
from haskie.indexing.embed import device_name
from haskie.settings import (
    PROFILES,
    RERANKER_MODELS,
    RERANKERS,
    Accelerator,
    Chunker,
    EmbeddingModel,
    EmbeddingProfile,
    FieldDoc,
    Fusion,
    ModelCard,
    Parser,
    Reranker,
    SearchMode,
    SearchSettings,
    UserSettings,
    docs,
    init_user_settings,
    load_user_settings,
    load_user_settings_or_none,
    save_user_settings,
    settings_problem,
)


class Status(msgspec.Struct):
    initialized: bool
    home: str
    embedding: EmbeddingModel | None
    device: str  # ONNX Runtime provider models run on (e.g. CoreML, CUDA, CPU)
    models: list[models.ModelStatus]  # download/load state of every model the settings need
    settings_error: str | None = None  # stored settings unreadable; defaults are in use


class Init(msgspec.Struct):
    """The first run's choices: what to embed with, and how to search. Everything else starts at
    its default and is changed in the settings later."""

    profile: EmbeddingProfile
    search: SearchSettings = msgspec.field(default_factory=SearchSettings)


class Options(msgspec.Struct):
    """Every choice the UI offers, from the constants the backend already validates against, so
    nothing is spelled a second time in the frontend."""

    parsers: tuple[Parser, ...]
    chunkers: tuple[Chunker, ...]
    accelerators: tuple[Accelerator, ...]
    search_modes: tuple[SearchMode, ...]
    fusions: tuple[Fusion, ...]
    rerankers: tuple[Reranker, ...]
    reranker_models: tuple[str, ...]
    reranker_cards: dict[str, ModelCard]  # every reranker model, offered here or not
    docs: dict[str, FieldDoc]  # title + definition per setting key, e.g. "conversion.chunk_size"
    embedding_profiles: dict[EmbeddingProfile, EmbeddingModel | None]
    document_statuses: tuple[DocumentStatus, ...]
    # in the import pipeline: a poll waits on them
    active_document_statuses: tuple[DocumentStatus, ...]
    member_statuses: tuple[MemberStatus, ...]
    active_member_statuses: tuple[MemberStatus, ...]
    active_run_statuses: tuple[RunStatus, ...]  # run statuses that are still on their way
    operation_kinds: tuple[operations.OperationKind, ...]  # the order the Operations view shows
    bulk_kinds: tuple[operations.BulkKind, ...]  # the operations a 202 points at


def _changed_fields(before: msgspec.Struct, after: msgspec.Struct, prefix: str = "") -> list[str]:
    """Dotted names of the settings that differ, for the audit trail. Names only: a value may
    be a path or a model id, which does not belong in an audit record."""
    changed: list[str] = []
    for field in msgspec.structs.fields(after):
        old, new = getattr(before, field.name), getattr(after, field.name)
        if old == new:
            continue
        if isinstance(old, msgspec.Struct) and isinstance(new, msgspec.Struct):
            changed.extend(_changed_fields(old, new, f"{prefix}{field.name}."))
        else:
            changed.append(f"{prefix}{field.name}")
    return changed


@get("/api/status")
async def get_status() -> Status:
    saved = await load_user_settings_or_none()
    current = saved or UserSettings()
    return Status(
        initialized=saved is not None,
        home=str(home.HOME),
        embedding=current.embedding_model if saved else None,
        device=device_name(current.pipeline.accelerator),
        models=(await models.model_statuses()) if saved else [],
        settings_error=settings_problem(),
    )


@post("/api/init")
@audit.audited("settings.init")
async def post_init(data: Init) -> UserSettings:
    """First run: pick the embedding profile, the search mode and the reranker. The models
    download in the background; poll /api/status -> models."""
    audit.attach(profile=data.profile)
    settings = UserSettings(embedding=data.profile, search=data.search)
    if not await init_user_settings(settings):
        raise Conflict("already initialized; change embedding via settings and reindex")
    await workflows.apply_settings(settings)
    return settings


@get("/api/settings")
async def get_settings() -> UserSettings:
    return await load_user_settings()


@put("/api/settings")
@audit.audited("settings.update")
async def put_settings(data: UserSettings) -> UserSettings:
    """Saves, then applies: every queue limit follows its setting, new models start downloading.
    Values out of range are rejected while the body is decoded, so nothing invalid is stored."""
    changed = _changed_fields(await load_user_settings(), data)
    if changed:
        audit.attach(changed=",".join(changed))
    settings = await save_user_settings(data)
    await workflows.apply_settings(settings)
    return settings


OPTIONS = Options(
    parsers=tuple(Parser),
    chunkers=tuple(Chunker),
    accelerators=tuple(Accelerator),
    search_modes=tuple(SearchMode),
    fusions=tuple(Fusion),
    rerankers=tuple(Reranker),
    # an MLX model is offered only where it can load; one already chosen still validates
    reranker_models=tuple(name for name in RERANKER_MODELS if mlx_models.loadable(name)),
    reranker_cards=RERANKERS,
    docs=docs(),
    embedding_profiles={
        profile: model
        for profile, model in PROFILES.items()
        if model is None or mlx_models.loadable(model.name)
    },
    document_statuses=DOCUMENT_STATUSES,
    active_document_statuses=ACTIVE_DOCUMENT_STATUSES,
    member_statuses=MEMBER_STATUSES,
    active_member_statuses=ACTIVE_MEMBER_STATUSES,
    active_run_statuses=tuple(RunStatus(status) for status in ACTIVE_STATUS),
    operation_kinds=operations.KIND_ORDER,
    bulk_kinds=operations.BULK_KINDS,
)


@get("/api/options")
async def get_options() -> Options:
    return OPTIONS
