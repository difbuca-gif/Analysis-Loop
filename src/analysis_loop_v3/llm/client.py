"""OpenAI 호환 및 native Ollama LLM 클라이언트."""

from __future__ import annotations

import inspect
import json
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

DEFAULT_MAX_TOKENS = 4096
# 호출부가 timeout을 안 줬을 때의 기본값과, 어떤 경우에도 넘지 않는 상한.
DEFAULT_READ_TIMEOUT = 600.0
MAX_READ_TIMEOUT = 600.0
# 도구 호출 루프의 최대 라운드 수.
MAX_TOOL_ROUNDS = 6

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

ToolExecutor = Callable[[str, dict[str, Any]], str | Awaitable[str]]
EventSink = Callable[[str, dict[str, Any]], None]
def normalize_llm_usage(
    body: dict[str, Any], *, native_ollama: bool, model: str,
) -> dict[str, Any]:
    usage = body.get('usage') or {}
    if native_ollama:
        prompt_tokens = body.get('prompt_eval_count')
        completion_tokens = body.get('eval_count')
        finish_reason = body.get('done_reason')
        provider = 'ollama'
    else:
        prompt_tokens = usage.get('prompt_tokens')
        completion_tokens = usage.get('completion_tokens')
        choices = body.get('choices') or []
        finish_reason = choices[0].get('finish_reason') if choices else None
        provider = 'openai-compatible'

    total_tokens = usage.get('total_tokens')
    if total_tokens is None and isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        total_tokens = prompt_tokens + completion_tokens

    values = {
        'provider': provider,
        'model': model,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'total_tokens': total_tokens,
        'finish_reason': finish_reason,
    }
    return {key: value for key, value in values.items() if value is not None}

class LLMError(RuntimeError):
    pass

class LLMTransportError(LLMError):
    """endpoint에 연결하지 못했거나 제한 시간 안에 응답을 받지 못했다."""

class LLMProviderError(LLMError):
    """endpoint가 HTTP 오류를 반환했다."""

    def __init__(self, message: str, *, status_code: int, retryable: bool) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable

def parse_llm_json(text: str) -> dict[str, Any]:
    """모델 응답에서 JSON 객체를 뽑는다.

    <think> 제거를 남기는 이유: 파서 없이 띄운 엔진도 같은 클라이언트를 쓴다.
    """
    cleaned = _THINK_BLOCK.sub("", text or "").strip()
    fenced = _JSON_FENCE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    if not cleaned:
        raise LLMError("응답이 비어 있다")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # 앞뒤에 산문이 붙은 경우를 위해 가장 바깥 중괄호만 잘라 다시 시도한다.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMError(f"JSON을 찾을 수 없다: {cleaned[:200]}") from None
        try:
            parsed = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"JSON 파싱 실패: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError(f"최상위가 객체가 아니다: {type(parsed).__name__}")
    return parsed

class OpenAICompatClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        disable_thinking: bool = True,
        use_native_ollama: bool = False,
        native_think: bool = True,
        bounded: bool = True,
        num_ctx: int = 65536,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        # Ollama에서 thinking 제어가 필요하면 native API를 사용한다.
        self.use_native_ollama = use_native_ollama
        # 분석 방향 판단 능력을 보존하기 위해 thinking을 기본 활성화한다.
        self.native_think = native_think
        # 원격 유료 API(NVIDIA)만 응답 지연에 타임아웃을 건다. 로컬 자체 호스팅
        # 모델(Ollama, vLLM/SGLang)은 사용자가 직접 통제하는 하드웨어라 느려도
        # 임의 상한으로 죽이지 않는다. 죽일지는 사람이 판단한다.
        self.bounded = bounded
        # native Ollama 전용. vLLM/SGLang·NVIDIA는 서버가 자체 context를 관리한다.
        self.num_ctx = num_ctx
        self._client: httpx.AsyncClient | None = None
        # 관측용. 상태가 아니므로 그래프에 넣지 않는다.
        self.calls: list[dict[str, Any]] = []
        self.event_sink: EventSink | None = None

    def _emit_call(
        self, name: str, *, role: str, started: float, **payload: Any,
    ) -> None:
        if self.event_sink is None:
            return
        record = {
            'role': role,
            'latency_seconds': round(time.monotonic() - started, 3),
            **{key: value for key, value in payload.items() if value is not None},
        }
        try:
            self.event_sink(name, record)
        except Exception:
            pass

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _read_timeout(self, requested: float | None) -> float | None:
        if requested is None:
            return DEFAULT_READ_TIMEOUT if self.bounded else None
        if requested <= 0:
            raise LLMError("LLM timeout은 0보다 커야 한다")
        return min(float(requested), MAX_READ_TIMEOUT) if self.bounded else float(requested)

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=MAX_READ_TIMEOUT)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @property
    def _provider_name(self) -> str:
        return 'ollama' if self.use_native_ollama else 'openai-compatible'

    def _fail(self, role: str, started: float, error_type: str, **extra: Any) -> None:
        """실패를 이벤트로 남긴다. 호출부가 곧바로 예외를 올린다."""
        self._emit_call(
            'LLM_CALL_FAILED', role=role, started=started,
            provider=self._provider_name, model=self.model,
            error_type=error_type, **extra,
        )

    def _build_request(
        self, *, messages: list[dict[str, Any]], json_mode: bool,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str, dict[str, Any]]:
        """(url, payload). 두 백엔드는 요청 필드가 아예 다르다."""
        if self.use_native_ollama:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": self.native_think,
                "options": {
                    "temperature": self.temperature,
                    # -1은 Ollama가 문서화한 "제한 없음"으로 EOS나 context 한계까지
                    # 생성한다. 로컬 자체 호스팅은 응답이 길어도 임의 상한으로 잘라
                    # 재파싱 실패를 만들지 않는다. 원격 유료 API만 max_tokens를 건다.
                    "num_predict": self.max_tokens if self.bounded else -1,
                    # 누적 문맥을 수용할 수 있도록 Ollama context를 명시한다.
                    "num_ctx": self.num_ctx,
                },
            }
            if json_mode:
                payload["format"] = "json"
            if tools:
                payload["tools"] = tools
            return f"{self.base_url.removesuffix('/v1')}/api/chat", payload

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.bounded:
            # 이 셋이 272초 -> 5.6초를 만든 파라미터다. 지우지 말 것.
            payload["max_tokens"] = self.max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return f"{self.base_url}/chat/completions", payload

    def _extract_message(
        self, body: Any, role: str, started: float,
    ) -> tuple[dict[str, Any], str | None]:
        """(assistant message, 종료사유). 응답 모양도 백엔드마다 다르다."""
        try:
            if self.use_native_ollama:
                message = body["message"]
                done_reason = body.get("done_reason")
            else:
                choice = body["choices"][0]
                message = choice["message"]
                done_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            self._fail(role, started, 'MalformedResponse')
            raise LLMError(f"예상과 다른 응답 구조: {str(body)[:300]}") from exc

        if not isinstance(message, dict):
            self._fail(role, started, 'MalformedResponse')
            raise LLMError(
                f"assistant message가 객체가 아니다: {type(message).__name__}"
            )
        return message, done_reason

    async def _post_chat_message(
        self, *, messages: list[dict[str, Any]], json_mode: bool,
        timeout_seconds: float | None,
        tools: list[dict[str, Any]] | None = None,
        role: str = 'unknown',
    ) -> tuple[dict[str, Any], str | None]:
        """공통 POST. use_native_ollama면 /api/chat, 아니면 /chat/completions."""
        client = await self._ensure_client()
        timeout = self._read_timeout(timeout_seconds)
        started = time.monotonic()
        url, payload = self._build_request(
            messages=messages, json_mode=json_mode, tools=tools,
        )

        try:
            response = await client.post(
                url, json=payload, headers=self._headers(), timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            self._fail(role, started, type(exc).__name__)
            raise LLMTransportError(
                f"{self.model}이 {timeout:.1f}초 안에 응답하지 않았다 (endpoint={url})"
            ) from exc
        except httpx.HTTPError as exc:
            self._fail(role, started, type(exc).__name__)
            raise LLMTransportError(
                f"{self.model} endpoint 통신 실패: {type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code >= 400:
            self._fail(role, started, 'HTTPError', status_code=response.status_code)
            raise LLMProviderError(
                f"{self.model} HTTP {response.status_code}: {response.text[:300]}",
                status_code=response.status_code,
                retryable=response.status_code in {408, 429} or response.status_code >= 500,
            )

        try:
            body = response.json()
        except ValueError as exc:
            self._fail(role, started, type(exc).__name__)
            raise LLMError(f"JSON이 아닌 API 응답: {response.text[:300]}") from exc

        message, done_reason = self._extract_message(body, role, started)
        self._emit_call(
            'LLM_CALL_COMPLETED', role=role, started=started,
            **normalize_llm_usage(
                body, native_ollama=self.use_native_ollama, model=self.model,
            ),
        )
        return message, done_reason

    async def _post_chat(
        self, *, messages: list[dict[str, Any]], json_mode: bool,
        timeout_seconds: float | None, role: str,
    ) -> tuple[str, str | None]:
        message, done_reason = await self._post_chat_message(
            messages=messages,
            json_mode=json_mode,
            timeout_seconds=timeout_seconds,
            role=role,
        )
        return str(message.get("content") or ""), done_reason

    def _raise_if_truncated(self, done_reason: str | None) -> None:
        # 길이 제한 종료는 JSON 파싱 전에 별도 오류로 처리한다.
        if done_reason == "length":
            raise LLMError(
                f"응답이 max_tokens({self.max_tokens})에서 잘렸다 — "
                "더 짧게 작성하거나 해당 역할의 _LLM_MAX_TOKENS를 올려라"
            )

    async def generate_json(
        self,
        *,
        prompt: str,
        system: str | None = None,
        timeout_seconds: float | None = None,
        role: str = "unknown",
    ) -> dict[str, Any]:
        content, done_reason = await self._post_chat(
            messages=self._messages(prompt, system), json_mode=True,
            timeout_seconds=timeout_seconds,
            role=role,
        )
        self._raise_if_truncated(done_reason)
        self.calls.append({"role": role, "model": self.model, "chars": len(content or "")})
        return parse_llm_json(content)

    @staticmethod
    def _parse_one_call(index: int, item: Any) -> tuple[str, str, dict[str, Any]]:
        """(call_id, name, arguments). 구조가 어긋나면 LLMError."""
        if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
            raise LLMError("예상과 다른 tool call 구조")
        function = item["function"]
        name = str(function.get("name") or "").strip()
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise LLMError(f"{name or 'tool'} 인자 JSON 파싱 실패: {exc}") from exc
        if not name or not isinstance(arguments, dict):
            raise LLMError("tool call의 name 또는 arguments가 올바르지 않다")
        return str(item.get("id") or f"call_{index}"), name, arguments

    @classmethod
    def _parse_tool_calls(
        cls, message: dict[str, Any],
    ) -> list[tuple[str, str, dict[str, Any]]]:
        return [
            cls._parse_one_call(index, item)
            for index, item in enumerate(message.get("tool_calls") or [])
        ]

    @staticmethod
    def _echo_assistant(message: dict[str, Any], content: str) -> dict[str, Any]:
        """모델이 방금 한 말을 대화에 되돌려 넣는다. thinking과 tool_calls는 있을 때만."""
        echo: dict[str, Any] = {"role": "assistant", "content": content}
        for key in ("thinking", "tool_calls"):
            if message.get(key):
                echo[key] = message[key]
        return echo

    def _tool_message(self, call_id: str, name: str, result: str) -> dict[str, Any]:
        """도구 결과를 대화에 넣는다. Ollama는 tool_name, OpenAI는 tool_call_id를 쓴다."""
        if self.use_native_ollama:
            return {"role": "tool", "tool_name": name, "content": result}
        return {"role": "tool", "tool_call_id": call_id, "content": result}

    @staticmethod
    async def _run_tool(execute_tool: ToolExecutor, name: str, arguments: dict[str, Any]) -> str:
        """도구 실패는 예외로 올리지 않고 결과 문자열로 돌려준다. 모델이 읽고
        고쳐 쓸 기회를 준다."""
        try:
            result = execute_tool(name, arguments)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"

    async def generate_json_with_tools(
        self,
        *,
        prompt: str,
        tools: list[dict[str, Any]],
        execute_tool: ToolExecutor,
        system: str | None = None,
        timeout_seconds: float | None = None,
        role: str = "unknown",
    ) -> dict[str, Any]:
        """모델이 도구 사용을 끝내고 반환한 최종 JSON.

        도구 호출이 반복될 수 있어 전체 라운드 수를 별도로 제한한다.
        """
        messages: list[dict[str, Any]] = list(self._messages(prompt, system))

        for _ in range(MAX_TOOL_ROUNDS):
            message, done_reason = await self._post_chat_message(
                messages=messages, json_mode=False,
                timeout_seconds=timeout_seconds, tools=tools, role=role,
            )
            self._raise_if_truncated(done_reason)
            tool_calls = self._parse_tool_calls(message)
            content = str(message.get("content") or "")
            self.calls.append({
                "role": role, "model": self.model,
                "chars": len(content), "tool_calls": len(tool_calls),
            })
            messages.append(self._echo_assistant(message, content))

            if not tool_calls:
                return parse_llm_json(content)

            for call_id, name, arguments in tool_calls:
                result = await self._run_tool(execute_tool, name, arguments)
                messages.append(self._tool_message(call_id, name, result))

        raise LLMError(
            f"tool 호출 루프가 {MAX_TOOL_ROUNDS}라운드를 넘었다 — "
            "모델이 도구 사용을 마치고 최종 JSON을 반환하지 않았다"
        )

    async def generate_text(
        self,
        *,
        prompt: str,
        system: str | None = None,
        timeout_seconds: float | None = None,
        role: str = "unknown",
    ) -> str:
        content, done_reason = await self._post_chat(
            messages=self._messages(prompt, system), json_mode=False,
            timeout_seconds=timeout_seconds,
            role=role,
        )
        self._raise_if_truncated(done_reason)
        self.calls.append({"role": role, "model": self.model, "chars": len(content or "")})
        return str(content or "")

def build_client(*, provider: str, model: str, base_url: str | None) -> OpenAICompatClient | None:
    provider = provider.lower()
    if provider == "none":
        return None
    if provider == "nvidia":
        api_key = os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            return None
        return OpenAICompatClient(
            model=model or "meta/llama-3.3-70b-instruct",
            base_url=base_url or NVIDIA_BASE_URL,
            api_key=api_key,
            # 외부 API는 자체 템플릿을 쓴다. vLLM 전용 확장을 보내면 400이 난다.
            disable_thinking=False,
            # 원격 유료 API라 응답 지연에 상한을 둔다.
            bounded=True,
        )
    if provider == "ollama":
        # native API에서만 Ollama thinking 옵션을 명시적으로 제어할 수 있다.
        return OpenAICompatClient(
            model=model or "qwen2.5:14b",
            base_url=base_url or "http://localhost:11434/v1",
            use_native_ollama=True,
            # 로컬 자체 호스팅 — 느려도 임의 타임아웃으로 죽이지 않는다.
            bounded=False,
        )
    # local: vLLM / SGLang. 자체 호스팅이라 API 키가 필요 없다.
    return OpenAICompatClient(
        model=model or "Qwen/Qwen3.5-9B",
        base_url=base_url or "http://localhost:8000/v1",
        # 이 경로도 로컬 자체 호스팅이다.
        bounded=False,
    )
