"""Small synchronous Streamable HTTP MCP client, compatible with Python 3.9.

The fixed anonymous endpoint needs no SDK, API key, embedding model or vector DB.
MCP connection IDs and conversation IDs deliberately have separate owners.
"""
import json
import re
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4


ENDPOINT = "https://search.parallel.ai/mcp"
USER_AGENT = "LLM-As-Chatbot/Parallel-Search (+https://github.com/deep-diver/LLM-As-Chatbot)"
PROTOCOL_VERSION = "2025-03-26"
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_CONTEXT_CHARS = 4000


class ParallelSearchError(RuntimeError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ParallelSearchError("Parallel Search MCP unexpectedly redirected the request.")


class ParallelSearchClient:
    def __init__(self, timeout=20):
        self.timeout = timeout
        self._connection_id = None
        self._opener = build_opener(_NoRedirect())

    def _request(self, method, params, notification=False):
        request_id = str(uuid4())
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notification:
            payload["id"] = request_id
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if self._connection_id:
            headers["Mcp-Session-Id"] = self._connection_id
        request = Request(ENDPOINT, json.dumps(payload).encode("utf-8"), headers)
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                connection_id = response.headers.get("Mcp-Session-Id")
                if method == "initialize":
                    self._connection_id = connection_id
                if notification:
                    return None
                if response.headers.get_content_type() == "text/event-stream":
                    message = self._read_event(response, request_id, monotonic() + self.timeout)
                else:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise ParallelSearchError("Parallel Search MCP response is too large.")
                    message = json.loads(raw)
        except HTTPError as error:
            status = error.code
            error.close()
            if status == 429:
                raise ParallelSearchError("Parallel Search MCP quota reached. Try again later.") from None
            raise ParallelSearchError("Parallel Search MCP HTTP error: " + str(status)) from None
        except (URLError, TimeoutError, OSError) as error:
            raise ParallelSearchError("Parallel Search MCP could not complete the request: " + str(error)) from error
        except (ValueError, UnicodeError) as error:
            raise ParallelSearchError("Parallel Search MCP returned an invalid response.") from error
        if not isinstance(message, dict) or message.get("id") != request_id:
            raise ParallelSearchError("Parallel Search MCP returned an unexpected response ID.")
        if "error" in message:
            raise ParallelSearchError("Parallel Search MCP rejected the request: " + str(message["error"])[:400])
        result = message.get("result")
        if not isinstance(result, dict):
            raise ParallelSearchError("Parallel Search MCP returned no result.")
        return result

    @staticmethod
    def _read_event(response, request_id, deadline):
        data = []
        total = 0
        while True:
            if monotonic() >= deadline:
                raise ParallelSearchError("Parallel Search MCP stream timed out.")
            line = response.readline(MAX_RESPONSE_BYTES + 1)
            total += len(line)
            if total > MAX_RESPONSE_BYTES:
                raise ParallelSearchError("Parallel Search MCP response is too large.")
            if not line or line in (b"\n", b"\r\n"):
                if data:
                    message = json.loads("\n".join(data))
                    if isinstance(message, dict) and message.get("id") == request_id:
                        return message
                    data = []
                if not line:
                    raise ParallelSearchError("Parallel Search MCP stream ended without a result.")
            elif line.startswith(b"data:"):
                data.append(line[5:].decode("utf-8").strip())

    def connect(self):
        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "LLM-As-Chatbot", "version": "1.0"},
        })
        if result.get("protocolVersion") != PROTOCOL_VERSION:
            raise ParallelSearchError("Parallel Search MCP negotiated an unsupported protocol version.")
        self._request("notifications/initialized", {}, notification=True)
        tools = self._request("tools/list", {}).get("tools", [])
        names = {tool.get("name") for tool in tools if isinstance(tool, dict)}
        if not {"web_search", "web_fetch"}.issubset(names):
            raise ParallelSearchError("Parallel Search MCP is missing search or fetch tools.")

    def call(self, name, arguments, session_id):
        result = self._request("tools/call", {
            "name": name,
            "arguments": {**arguments, "session_id": session_id},
        })
        if result.get("isError"):
            messages = [block.get("text", "") for block in result.get("content", [])
                        if isinstance(block, dict) and block.get("type") == "text"]
            raise ParallelSearchError("Parallel Search MCP: " + " ".join(messages)[:400])
        content = result.get("structuredContent")
        if not isinstance(content, dict):
            for block in result.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    try:
                        content = json.loads(block.get("text", ""))
                    except ValueError:
                        continue
                    if isinstance(content, dict) and "results" in content:
                        break
        if not isinstance(content, dict) or not isinstance(content.get("results"), list):
            raise ParallelSearchError("Parallel Search MCP returned invalid tool content.")
        return content


def _urls(question):
    urls = []
    for value in re.findall(r'https?://[^\s<>"\]]+', question):
        value = value.rstrip(".,;:!?")
        while value.endswith(")") and value.count(")") > value.count("("):
            value = value[:-1]
        try:
            parsed = urlsplit(value)
        except ValueError:
            continue
        if parsed.hostname and not parsed.username and not parsed.password and value not in urls:
            urls.append(value)
    return urls[:20]


def search_context(question, session_id):
    client = ParallelSearchClient()
    client.connect()
    urls = _urls(question)
    if urls:
        content = client.call("web_fetch", {"urls": urls, "objective": question[:200]}, session_id)
    else:
        content = client.call("web_search", {"objective": question, "search_queries": [question]}, session_id)
    sources = []
    for result in content["results"]:
        if not isinstance(result, dict):
            continue
        url = result.get("url", "")
        excerpts = result.get("excerpts", [])
        if not isinstance(url, str):
            continue
        try:
            parsed = urlsplit(url)
        except ValueError:
            continue
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            continue
        if not isinstance(excerpts, list) or not any(isinstance(t, str) and t.strip() for t in excerpts):
            continue
        title = result.get("title") or url
        sources.append(str(title) + "\n" + url + "\n" + "\n".join(t for t in excerpts if isinstance(t, str)))
    if not sources:
        raise ParallelSearchError("Parallel Search MCP returned no usable web excerpts. Try a different query or URL.")
    # Bound retrieval context for the small local models supported by this app.
    return "\n\n".join(sources)[:MAX_CONTEXT_CHARS]
