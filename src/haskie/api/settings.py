"""Init, user settings and the option catalogue the UI renders its forms from."""

import msgspec
from litestar import get, post, put

from haskie import audit, home, models, workflows
from haskie.errors import Conflict
from haskie.settings import (
    ACCELERATORS,
    CHUNKERS,
    FUSIONS,
    PARSERS,
    PROFILES,
    RERANKER_MODELS,
    RERANKERS,
    SEARCH_MODES,
    EmbeddingModel,
    EmbeddingProfile,
    FieldDoc,
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
    profile: EmbeddingProfile


class Options(msgspec.Struct):
    parsers: tuple[str, ...]
    chunkers: tuple[str, ...]
    accelerators: tuple[str, ...]
    search_modes: tuple[str, ...]
    fusions: tuple[str, ...]
    rerankers: tuple[str, ...]
    reranker_models: tuple[str, ...]
    docs: dict[str, FieldDoc]  # title + definition per setting key, e.g. "defaults.chunk_size"
    embedding_profiles: dict[str, EmbeddingModel | None]


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
    from haskie.embed import device_name

    saved = await load_user_settings_or_none()
    current = saved or UserSettings()
    return Status(
        initialized=saved is not None,
        home=str(home.HOME),
        embedding=current.embedding_model if saved else None,
        device=device_name(current.pipeline.accelerator),
        models=(await models.model_statuses(current)) if saved else [],
        settings_error=settings_problem(),
    )


@post("/api/init")
@audit.audited("settings.init")
async def post_init(data: Init) -> UserSettings:
    """First run: pick the embedding profile. The model downloads in the background; poll
    /api/status -> models."""
    audit.attach(profile=data.profile)
    settings = UserSettings(embedding=data.profile)
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
    parsers=PARSERS,
    chunkers=CHUNKERS,
    accelerators=ACCELERATORS,
    search_modes=SEARCH_MODES,
    fusions=FUSIONS,
    rerankers=RERANKERS,
    reranker_models=RERANKER_MODELS,
    docs=docs(),
    embedding_profiles=PROFILES,
)


@get("/api/options")
async def get_options() -> Options:
    return OPTIONS
