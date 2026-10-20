"""Conversation identity for Gradio's separately serialized chat channels."""
from contextvars import ContextVar
from uuid import uuid4


current_search_session = ContextVar("parallel_search_session", default=None)


def search_session(state, index):
    # Gradio owns state per browser session; channels have stable numeric indices.
    sessions = state.setdefault("parallel_search_sessions", {})
    key = str(index)
    if key not in sessions:
        sessions[key] = str(uuid4())
    return sessions[key]


def reset_search_session(state, index):
    state.get("parallel_search_sessions", {}).pop(str(index), None)


def search_session_stream(stream, session_id):
    """Bind only while advancing a model generator, never across a yielded UI update."""
    try:
        while True:
            token = current_search_session.set(session_id)
            try:
                item = next(stream)
            except StopIteration:
                return
            finally:
                current_search_session.reset(token)
            yield item
    finally:
        stream.close()
