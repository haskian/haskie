"""The HTTP/MCP surface, one module per feature.

`app.py` owns the application: the request context, the error mapping and `create_app`. Each
module here owns one feature's routes and the request/response shapes only that feature uses;
what two of them share lives in `common.py`.
"""

from litestar.types import ControllerRouterHandler

from haskie.api import documents, jobs, libraries, search, settings

# Order is the order the routes appear in the OpenAPI document, so it follows the UI: set up,
# then libraries and their documents, then the work, then searching.
ROUTE_HANDLERS: list[ControllerRouterHandler] = [
    settings.get_status,
    settings.post_init,
    settings.get_settings,
    settings.put_settings,
    settings.get_options,
    libraries.list_libraries,
    libraries.create_library,
    libraries.get_library,
    libraries.list_documents,
    libraries.delete_library,
    libraries.search_library,
    libraries.put_library_settings,
    libraries.index_library,
    libraries.describe_library,
    documents.upload_document,
    documents.import_document,
    documents.index_document,
    documents.delete_document,
    documents.get_source,
    documents.get_preview,
    documents.get_markdown,
    documents.describe_document,
    jobs.list_jobs,
    jobs.list_jobs_by_kind,
    jobs.list_job_kinds,
    jobs.get_activity,
    jobs.list_job_tasks,
    jobs.get_job_progress,
    jobs.delete_job,
    search.list_sessions,
    search.put_session,
    search.search_session,
    search.search_text,
    search.search_documents,
]
