import asyncio
import io
import json
import unittest
from email.message import Message
from urllib.error import HTTPError
from unittest.mock import patch

from chats import parallel_search
from chats.search_sessions import current_search_session
from tests.runtime_fixture import runtime, state_and_history, chat_arguments


class Response(io.BytesIO):
    def __init__(self, data, content_type="application/json"):
        super().__init__(data)
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        self.headers["Mcp-Session-Id"] = "transport-session"


class Endpoint:
    def __init__(self, sse=False, text_only=False, failure=None, results=None):
        self.requests = []
        self.responses = []
        self.sse = sse
        self.text_only = text_only
        self.failure = failure
        self.results = results if results is not None else [{
            "url": "https://www.python.org/", "title": "Python",
            "excerpts": ["Python is a programming language."],
        }]

    def open(self, request, timeout):
        payload = json.loads(request.data)
        self.requests.append((request, payload, timeout))
        method = payload["method"]
        if method == "initialize":
            result = {"protocolVersion": parallel_search.PROTOCOL_VERSION}
        elif method == "notifications/initialized":
            response = Response(b"")
            self.responses.append(response)
            return response
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search"}, {"name": "web_fetch"}]}
        else:
            if isinstance(self.failure, BaseException):
                raise self.failure
            if self.failure:
                result = self.failure
            else:
                content = {"results": self.results}
                result = {"content": [{"type": "text", "text": json.dumps(content)}]}
                if not self.text_only:
                    result["structuredContent"] = content
        raw = json.dumps({"jsonrpc": "2.0", "id": payload["id"], "result": result}).encode()
        if self.sse:
            # A notification and comments precede the matching JSON-RPC result.
            raw = (b': keepalive\n\nevent: message\ndata: {"jsonrpc":"2.0","method":"notice"}\n\n'
                   + b"event: message\ndata: " + raw + b"\n\n")
        response = Response(raw, "text/event-stream" if self.sse else "application/json")
        self.responses.append(response)
        return response

    @property
    def calls(self):
        return [payload["params"] for _, payload, _ in self.requests if payload["method"] == "tools/call"]


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = runtime()
        cls.runtime = cls.fixture.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.__exit__(None, None, None)

    def setUp(self):
        self.runtime.prompts.clear()
        self.runtime.globals.remote_addr = None
        self.state, self.history = state_and_history()

    def stream(self, **kwargs):
        return self.runtime.central.sync_chat_stream(**chat_arguments(
            self.runtime.central, self.state, self.history, **kwargs))

    def test_fresh_no_key_actual_dispatcher_into_prompt_and_ui(self):
        endpoint = Endpoint()
        # Ambient credentials must never turn the anonymous path into a paid call.
        with patch.dict("os.environ", {"PARALLEL_API_KEY": "unused-test-key"}), \
                patch.object(parallel_search, "build_opener", return_value=endpoint):
            updates = list(self.stream())
        self.assertIn("Python is a programming language.", self.runtime.prompts[-1])
        self.assertIn("https://www.python.org/", self.runtime.prompts[-1])
        self.assertIn("Scripted model response.", str(updates[-1][1]))
        self.assertEqual(json.loads(updates[-1][3])[0]["pingpongs"][-1]["ping"], "What is Python?")
        self.assertEqual([p["method"] for _, p, _ in endpoint.requests],
                         ["initialize", "notifications/initialized", "tools/list", "tools/call"])
        for request, payload, timeout in endpoint.requests:
            self.assertEqual(request.full_url, "https://search.parallel.ai/mcp")
            headers = {key.lower(): value for key, value in request.header_items()}
            self.assertEqual(headers["user-agent"], parallel_search.USER_AGENT)
            self.assertNotIn("authorization", headers)
            self.assertEqual(timeout, 20)
            if payload["method"] != "initialize":
                self.assertEqual(headers["mcp-session-id"], "transport-session")
        self.assertTrue(all(response.closed for response in endpoint.responses))
        self.assertIsNone(current_search_session.get())

    def test_off_suppresses_all_search_even_with_a_key(self):
        with patch.object(parallel_search, "build_opener") as opener:
            list(self.stream(mode="off", key="configured-key"))
        opener.assert_not_called()
        self.assertNotIn("web excerpts", self.runtime.prompts[-1])

    def test_explicit_key_keeps_serper_and_never_falls_back(self):
        import sys
        import types
        from unittest.mock import Mock
        legacy = types.ModuleType("pingpong.context")
        legacy.SimilaritySearcher = Mock()
        legacy.InternetSearchStrategy = Mock(side_effect=RuntimeError("Serper rejected the key"))
        with patch.dict(sys.modules, {"pingpong.context": legacy}), \
                patch.object(parallel_search, "build_opener") as opener:
            with self.assertRaisesRegex(RuntimeError, "Serper rejected"):
                list(self.stream(key="configured-key"))
        self.assertEqual(legacy.InternetSearchStrategy.call_args.kwargs["serper_api_key"], "configured-key")
        opener.assert_not_called()
        self.assertEqual(self.runtime.prompts, [])

    def test_configured_serper_result_reaches_the_model(self):
        import copy
        import sys
        import types
        from unittest.mock import Mock

        def retrieve(ppmanager):
            result = copy.deepcopy(ppmanager)
            result.pingpongs[-1].ping += "\nExisting Serper retrieval."
            yield result, "done"

        legacy = types.ModuleType("pingpong.context")
        legacy.SimilaritySearcher = Mock()
        legacy.InternetSearchStrategy = Mock(return_value=retrieve)
        with patch.dict(sys.modules, {"pingpong.context": legacy}), \
                patch.object(parallel_search, "build_opener") as opener:
            list(self.stream(key="configured-key"))
        opener.assert_not_called()
        self.assertIn("Existing Serper retrieval.", self.runtime.prompts[-1])

    def test_followup_fetch_independent_channels_interleaving_and_reset(self):
        endpoint = Endpoint()
        with patch.object(parallel_search, "build_opener", return_value=endpoint):
            first = self.stream(index=0)
            second = self.stream(index=1)
            next(first)
            next(second)
            next(first)
            next(second)
            first_updates = list(first)
            list(second)
            self.history = json.loads(first_updates[-1][3])
            fetch_updates = list(self.stream(index=0, question="Read https://www.python.org/"))
            self.history = json.loads(fetch_updates[-1][3])
            list(self.stream(index=0, question="What else about Python?"))
            ids = [call["arguments"]["session_id"] for call in endpoint.calls]
            self.assertNotEqual(ids[0], ids[1])
            self.assertEqual(ids[0], ids[2])
            self.assertEqual(ids[0], ids[3])
            self.assertNotEqual(ids[0], "transport-session")
            self.assertEqual(endpoint.calls[2]["name"], "web_fetch")
            # Execute the actual reset handler. Only its UI/model imports are
            # stand-ins; chat storage and PingPong serialization remain real.
            import importlib
            import sys
            import types
            gradio = types.SimpleNamespace(SelectData=object, update=lambda **kwargs: kwargs)
            transformers = types.SimpleNamespace(AutoModelForCausalLM=object)
            with patch.dict(sys.modules, {"gradio": gradio, "transformers": transformers}):
                app = importlib.import_module("app")
                reset = app.reset_chat(0, self.history, self.state)
                self.history = json.loads(reset[2])
            self.assertEqual(self.history[0]["pingpongs"], [])
            list(self.stream(index=0))
            self.assertNotEqual(ids[0], endpoint.calls[-1]["arguments"]["session_id"])
            other_state, _ = state_and_history()
            args = chat_arguments(self.runtime.central, other_state, self.history)
            list(self.runtime.central.sync_chat_stream(**args))
            self.assertNotEqual(ids[0], endpoint.calls[-1]["arguments"]["session_id"])
        self.assertIsNone(current_search_session.get())

    def test_sse_and_text_content_are_consumed(self):
        endpoint = Endpoint(sse=True, text_only=True)
        with patch.object(parallel_search, "build_opener", return_value=endpoint):
            list(self.stream())
        self.assertIn("Python is a programming language.", self.runtime.prompts[-1])

    def test_errors_timeouts_and_retry_keep_identity_without_generation(self):
        for failure in [TimeoutError("socket timed out"),
                        HTTPError(parallel_search.ENDPOINT, 429, "quota", {}, io.BytesIO()),
                        {"isError": True, "content": [{"type": "text", "text": "quota reached"}]},
                        {"structuredContent": {"unexpected": "shape"}},
                        {"structuredContent": {"results": []}}]:
            with self.subTest(failure=type(failure).__name__):
                endpoint = Endpoint(failure=failure)
                with patch.object(parallel_search, "build_opener", return_value=endpoint):
                    with self.assertRaises(parallel_search.ParallelSearchError):
                        list(self.stream())
                    session_id = endpoint.calls[-1]["arguments"]["session_id"]
                    endpoint.failure = None
                    list(self.stream())
                    self.assertEqual(session_id, endpoint.calls[-1]["arguments"]["session_id"])
                self.runtime.prompts.clear()
                self.assertIsNone(current_search_session.get())

    def test_stop_before_request_and_generator_exit_release_context(self):
        with patch.object(parallel_search, "build_opener") as opener:
            stream = self.stream()
            next(stream)  # Progress is yielded before opening the connection.
            stream.close()
            opener.assert_not_called()
        for failure in [GeneratorExit(), asyncio.CancelledError()]:
            endpoint = Endpoint(failure=failure)
            with patch.object(parallel_search, "build_opener", return_value=endpoint):
                with self.assertRaises(type(failure)):
                    list(self.stream())
        self.assertIsNone(current_search_session.get())
        self.assertEqual(self.runtime.prompts, [])

    def test_malformed_wire_response_does_not_generate(self):
        for raw in [b"not json", b'{}', b'{"jsonrpc":"2.0","id":"wrong","result":{}}']:
            endpoint = Endpoint()
            original = endpoint.open

            def open_response(request, timeout):
                if json.loads(request.data)["method"] == "tools/call":
                    return Response(raw)
                return original(request, timeout)

            with patch.object(parallel_search, "build_opener", return_value=endpoint), \
                    patch.object(endpoint, "open", side_effect=open_response):
                with self.assertRaises(parallel_search.ParallelSearchError):
                    list(self.stream())
        self.assertEqual(self.runtime.prompts, [])

    def test_remote_tgi_consumes_retrieval_and_reuses_channel_identity(self):
        self.runtime.globals.remote_addr = "http://model.example"
        endpoint = Endpoint()
        prompts = []

        async def generate(prompt, **kwargs):
            prompts.append(prompt)
            yield "Scripted TGI response."

        async def run():
            args = chat_arguments(self.runtime.central, self.state, self.history)
            return [update async for update in self.runtime.central.chat_stream(**args)]

        with patch.object(self.runtime.central.remote_tgi, "gen_text", generate), \
                patch.object(parallel_search, "build_opener", return_value=endpoint):
            updates = asyncio.run(run())
            list(self.stream())
        self.assertIn("Python is a programming language.", prompts[0])
        self.assertIn("https://www.python.org/", prompts[0])
        self.assertIn("Scripted TGI response.", str(updates[-1][1]))
        self.assertEqual(endpoint.calls[0]["arguments"]["session_id"],
                         endpoint.calls[1]["arguments"]["session_id"])

    def test_no_usable_urls_or_excerpts_and_bounded_context(self):
        for results in [[], [{"url": "javascript:alert(1)", "excerpts": ["bad"]}],
                        [{"url": "https://example.com", "excerpts": []}]]:
            with patch.object(parallel_search, "build_opener", return_value=Endpoint(results=results)):
                with self.assertRaises(parallel_search.ParallelSearchError):
                    list(self.stream())
        endpoint = Endpoint(results=[{"url": "https://example.com", "excerpts": ["x" * 50000]}])
        with patch.object(parallel_search, "build_opener", return_value=endpoint):
            list(self.stream())
        self.assertLess(len(self.runtime.prompts[-1]), 5000)


if __name__ == "__main__":
    unittest.main()
