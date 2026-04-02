"""Agno model for the ChatGPT Codex backend using local OAuth."""

from __future__ import annotations

import asyncio
import json
import random
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Type, Union

import requests
from pydantic import BaseModel

from agno.exceptions import ModelProviderError
from agno.models.base import Model
from agno.models.message import Message
from agno.models.metrics import Metrics
from agno.models.response import ModelResponse
from agno.run.agent import RunOutput
from agno.run.team import TeamRunOutput

try:
    from agno.exceptions import ModelAuthenticationError
except ImportError:
    class ModelAuthenticationError(ModelProviderError):
        """Backward-compatible auth error for older Agno builds."""

        def __init__(
            self,
            message: str,
            status_code: int = 401,
            model_name: Optional[str] = None,
            model_id: Optional[str] = None,
        ):
            super().__init__(
                message=message,
                status_code=status_code,
                model_name=model_name,
                model_id=model_id,
            )

from valuecell.openai_oauth.auth import (
    decode_jwt_payload,
    extract_chatgpt_account_id,
    refresh_access_token,
)
from valuecell.openai_oauth.exceptions import CodexAPIError, NotAuthenticatedError
from valuecell.openai_oauth.instructions import get_codex_instructions, normalize_model
from valuecell.openai_oauth.store import AuthStore, OAuthCredentials

Role = Literal["developer", "user", "assistant"]
SystemPromptMode = Literal["strict", "default", "disabled"]
CODEX_BASE_URL = "https://chatgpt.com/backend-api"
CODEX_RESPONSES_PATH = "/codex/responses"
DEFAULT_INCLUDE = ["reasoning.encrypted_content"]


def _as_dict(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _backoff_seconds(attempt: int) -> float:
    base = min(8.0, 0.5 * (2**attempt))
    return base * (1.0 + random.random() * 0.1)


def _is_retryable_status(status_code: int | None) -> bool:
    return status_code in {429, 500, 502, 503, 504}


def _iter_sse_events(lines: Iterable[str | bytes]) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    for raw_line in lines:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
        else:
            line = raw_line.rstrip("\n")
        if not line:
            if data_lines:
                payload = "\n".join(data_lines)
                data_lines = []
                if payload.strip() == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield event
            continue

        if line.startswith(":"):
            continue

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        payload = "\n".join(data_lines)
        if payload.strip() != "[DONE]":
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                return
            if isinstance(event, dict):
                yield event


def _is_terminal_event(event: dict[str, Any]) -> bool:
    return str(event.get("type") or "") in {"response.done", "response.completed"}


def _extract_system_texts(messages: list[Message]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if message.role in {"system", "developer"}:
            text = message.get_content_string()
            if text:
                texts.append(text)
    return texts


def _build_extra_instructions(texts: list[str]) -> str | None:
    if not texts:
        return None
    joined = "\n\n".join(texts)
    if not joined:
        return None
    return (
        "### Conversation system prompt\n"
        "Treat the following system instructions as highest priority.\n\n"
        f"{joined}\n\n"
        "### End conversation system prompt"
    )


def _message_item(role: Role, text: str) -> dict[str, Any]:
    block_type = "output_text" if role == "assistant" else "input_text"
    return {
        "type": "message",
        "role": role,
        "content": [{"type": block_type, "text": text}],
    }


def _function_call_item(call_id: str, name: str, args: dict[str, Any] | str) -> dict[str, Any]:
    arguments = args if isinstance(args, str) else json.dumps(args, separators=(",", ":"))
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _function_call_output_item(call_id: str, output: Any) -> dict[str, Any]:
    output_text = output if isinstance(output, str) else json.dumps(output)
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output_text,
    }


def _normalize_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        function_data = dict(tool["function"])
        return {
            "type": "function",
            **{
                key: value
                for key, value in function_data.items()
                if key not in {"requires_confirmation", "external_execution", "approval_type"}
            },
        }
    return tool


def _normalize_tool_choice(tool_choice: Any | None) -> Any | None:
    if tool_choice is None:
        return None

    choice_dict = _as_dict(tool_choice)
    if choice_dict is not None:
        if choice_dict.get("type") == "function" and isinstance(choice_dict.get("function"), dict):
            name = choice_dict["function"].get("name")
            if isinstance(name, str) and name:
                return {"type": "function", "name": name}
        if choice_dict.get("type") == "function" and isinstance(choice_dict.get("name"), str):
            return choice_dict
        return choice_dict

    if isinstance(tool_choice, bool):
        return "required" if tool_choice else None

    if not isinstance(tool_choice, str):
        return tool_choice

    lowered = tool_choice.strip().lower()
    if lowered == "any":
        return "required"
    if lowered in {"auto", "none", "required"}:
        return lowered
    return {"type": "function", "name": tool_choice.strip()}


def _to_input_items(
    messages: list[Message],
    *,
    system_prompt_mode: SystemPromptMode = "strict",
    compress_tool_results: bool = False,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    if system_prompt_mode == "strict":
        messages_to_process = [
            message for message in messages if message.role not in {"system", "developer"}
        ]
    elif system_prompt_mode == "disabled":
        messages_to_process = [
            message for message in messages if message.role not in {"system", "developer"}
        ]
    else:
        messages_to_process = list(messages)

    for message in messages_to_process:
        if system_prompt_mode == "default" and message.role in {"system", "developer"}:
            content = message.get_content_string()
            if content:
                items.append(_message_item("developer", content))
            continue

        if message.role == "user":
            items.append(_message_item("user", message.get_content_string()))
            continue

        if message.role == "tool":
            tool_call_id = message.tool_call_id
            if isinstance(tool_call_id, str) and tool_call_id:
                tool_result = message.get_content(use_compressed_content=compress_tool_results)
                items.append(_function_call_output_item(tool_call_id, tool_result))
            continue

        assistant_text = message.get_content_string()
        if assistant_text:
            items.append(_message_item("assistant", assistant_text))

        tool_calls = message.tool_calls or []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function_data = tool_call.get("function")
            name = function_data.get("name") if isinstance(function_data, dict) else None
            arguments = function_data.get("arguments") if isinstance(function_data, dict) else None
            call_id = tool_call.get("id") or tool_call.get("call_id") or f"call_{uuid.uuid4().hex}"
            if not isinstance(name, str) or not name:
                continue
            if isinstance(arguments, str):
                try:
                    parsed_args = json.loads(arguments)
                except Exception:
                    parsed_args = arguments
            else:
                parsed_args = arguments if isinstance(arguments, dict) else {}
            items.append(_function_call_item(str(call_id), name, parsed_args))

    return items


def _parse_assistant_response(response: object) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {
            "content": str(response) if response is not None else "",
            "tool_calls": [],
        }

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                content = item.get("content")
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        block_type = block.get("type")
                        text = block.get("text")
                        if block_type in {"output_text", "text"} and isinstance(text, str):
                            text_parts.append(text)
            elif item_type == "function_call":
                call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
                name = item.get("name")
                arguments = item.get("arguments")
                if not isinstance(name, str) or not name:
                    continue
                if isinstance(arguments, str):
                    try:
                        parsed_args = json.loads(arguments)
                    except Exception:
                        parsed_args = {}
                    serialized_args = arguments
                elif isinstance(arguments, dict):
                    parsed_args = arguments
                    serialized_args = json.dumps(arguments, separators=(",", ":"))
                else:
                    parsed_args = {}
                    serialized_args = "{}"
                tool_calls.append(
                    {
                        "id": str(call_id),
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serialized_args,
                        },
                        "parsed_arguments": parsed_args,
                    }
                )

    if not text_parts:
        output_text = response.get("output_text")
        if isinstance(output_text, str):
            text_parts.append(output_text)

    return {
        "content": "".join(text_parts),
        "tool_calls": tool_calls,
    }


def _extract_response_metadata(response: object) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}

    metadata: dict[str, Any] = {}
    response_id = response.get("id")
    if isinstance(response_id, str) and response_id:
        metadata["response_id"] = response_id
    model = response.get("model")
    if isinstance(model, str) and model:
        metadata["model"] = model
    status = response.get("status")
    if isinstance(status, str) and status:
        metadata["status"] = status
    return metadata


def _extract_usage_metrics(response: object) -> Metrics | None:
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None

    def _as_int(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    input_tokens = _as_int(usage.get("input_tokens")) or _as_int(usage.get("prompt_tokens"))
    output_tokens = _as_int(usage.get("output_tokens")) or _as_int(usage.get("completion_tokens"))
    total_tokens = _as_int(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if input_tokens is None or output_tokens is None or total_tokens is None:
        return None
    return Metrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _response_format_instruction(
    response_format: Optional[Union[dict[str, Any], Type[BaseModel]]],
) -> str | None:
    if response_format is None:
        return None

    if response_format == {"type": "json_object"}:
        return "Return only a valid JSON object. Do not wrap the response in markdown fences."

    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        schema = json.dumps(response_format.model_json_schema(), ensure_ascii=False)
        return (
            "Return only valid JSON matching this schema. "
            f"Schema: {schema}"
        )

    if isinstance(response_format, dict):
        if response_format.get("type") == "json_schema":
            schema = json.dumps(response_format, ensure_ascii=False)
            return (
                "Return only valid JSON matching this schema. "
                f"Schema: {schema}"
            )
        if response_format.get("type") == "json_object":
            return "Return only a valid JSON object. Do not wrap the response in markdown fences."
    return None


class _OAuthBackendClient:
    def __init__(
        self,
        auth_store: AuthStore | None = None,
        *,
        base_url: str = CODEX_BASE_URL,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._store = auth_store or AuthStore()
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    @staticmethod
    def _headers(creds: OAuthCredentials) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {creds.access}",
            "chatgpt-account-id": creds.account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
            "Accept": "text/event-stream",
        }

    def _load_valid_credentials(self, session: requests.Session) -> OAuthCredentials:
        creds = self._store.load()
        now_ms = int(time.time() * 1000)
        if creds.expires > now_ms:
            return creds

        refreshed = refresh_access_token(refresh_token=creds.refresh, session=session)
        payload = decode_jwt_payload(refreshed.access)
        if not payload:
            raise NotAuthenticatedError(
                "Token refresh succeeded but the token is invalid; re-login required."
            )

        account_id = extract_chatgpt_account_id(payload)
        if not account_id:
            raise NotAuthenticatedError(
                "Failed to derive account id from refreshed token; re-login required."
            )

        new_creds = OAuthCredentials(
            access=refreshed.access,
            refresh=refreshed.refresh,
            expires=refreshed.expires_at_ms,
            account_id=account_id,
        )
        self._store.save(new_creds)
        return new_creds

    def stream_events(
        self,
        *,
        input_items: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        text_verbosity: str | None = None,
        include: list[str] | None = None,
        extra_instructions: str | None = None,
        response_format: Optional[Union[dict[str, Any], Type[BaseModel]]] = None,
    ) -> Iterator[dict[str, Any]]:
        request_body: dict[str, Any] = {
            "model": normalize_model(model),
            "store": False,
            "stream": True,
            "input": input_items,
            "include": list(DEFAULT_INCLUDE if include is None else include),
        }

        if tools is not None:
            request_body["tools"] = [_normalize_tool(tool) for tool in tools]
        if tool_choice is not None:
            request_body["tool_choice"] = _normalize_tool_choice(tool_choice)
        if temperature is not None:
            request_body["temperature"] = temperature
        if max_output_tokens is not None:
            request_body["max_output_tokens"] = max_output_tokens
        if reasoning_effort:
            request_body["reasoning"] = {"effort": reasoning_effort}

        text_config: dict[str, Any] = {}
        if text_verbosity:
            text_config["verbosity"] = text_verbosity
        if response_format is not None:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                text_config["format"] = {
                    "type": "json_schema",
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                }
            elif isinstance(response_format, dict):
                text_config["format"] = response_format
        if text_config:
            request_body["text"] = text_config

        format_instruction = _response_format_instruction(response_format)
        url = f"{self._base_url}{CODEX_RESPONSES_PATH}"
        with requests.Session() as session:
            creds = self._load_valid_credentials(session)
            base_instructions = get_codex_instructions(
                model=request_body["model"],
                session=session,
            )
            extras = [extra_instructions, format_instruction]
            combined_extra = "\n\n".join([extra for extra in extras if extra])
            request_body["instructions"] = (
                f"{base_instructions}\n\n{combined_extra}".strip()
                if combined_extra
                else base_instructions
            )

            instructions_extra_removed = False
            tool_choice_removed = False
            temperature_removed = False
            max_output_tokens_removed = False
            format_removed = False
            attempt = 0

            while True:
                response: requests.Response | None = None
                try:
                    response = session.post(
                        url,
                        headers=self._headers(creds),
                        json=request_body,
                        stream=True,
                        timeout=self._timeout_s,
                    )

                    if response.status_code >= 400:
                        error = self._to_api_error(response)
                        error_text = str(error).lower()
                        if (
                            not instructions_extra_removed
                            and combined_extra
                            and error.status_code == 400
                            and "instruction" in error_text
                        ):
                            request_body["instructions"] = base_instructions
                            instructions_extra_removed = True
                            continue
                        if (
                            not tool_choice_removed
                            and tool_choice is not None
                            and error.status_code == 400
                            and "tool_choice" in error_text
                        ):
                            request_body.pop("tool_choice", None)
                            tool_choice_removed = True
                            continue
                        if (
                            not temperature_removed
                            and temperature is not None
                            and error.status_code == 400
                            and "temperature" in error_text
                        ):
                            request_body.pop("temperature", None)
                            temperature_removed = True
                            continue
                        if (
                            not max_output_tokens_removed
                            and max_output_tokens is not None
                            and error.status_code == 400
                            and any(token in error_text for token in ("max_output_tokens", "max_tokens"))
                        ):
                            request_body.pop("max_output_tokens", None)
                            max_output_tokens_removed = True
                            continue
                        if (
                            not format_removed
                            and "text" in request_body
                            and isinstance(request_body["text"], dict)
                            and request_body["text"].get("format") is not None
                            and error.status_code == 400
                            and "format" in error_text
                        ):
                            request_body["text"].pop("format", None)
                            if not request_body["text"]:
                                request_body.pop("text", None)
                            format_removed = True
                            continue
                        if _is_retryable_status(error.status_code) and attempt < self._max_retries:
                            time.sleep(_backoff_seconds(attempt))
                            attempt += 1
                            continue
                        raise error

                    yield from _iter_sse_events(response.iter_lines(decode_unicode=True))
                    return
                except (requests.Timeout, requests.ConnectionError) as exc:
                    if attempt < self._max_retries:
                        time.sleep(_backoff_seconds(attempt))
                        attempt += 1
                        continue
                    raise CodexAPIError(
                        "Network error calling the ChatGPT Codex backend.",
                        status_code=None,
                    ) from exc
                finally:
                    if response is not None:
                        response.close()

    def complete_with_response(
        self,
        *,
        input_items: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
        text_verbosity: str | None = None,
        include: list[str] | None = None,
        extra_instructions: str | None = None,
        response_format: Optional[Union[dict[str, Any], Type[BaseModel]]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        last_response: dict[str, Any] | None = None
        for event in self.stream_events(
            input_items=input_items,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            text_verbosity=text_verbosity,
            include=include,
            extra_instructions=extra_instructions,
            response_format=response_format,
        ):
            if _is_terminal_event(event):
                response = event.get("response")
                if isinstance(response, dict):
                    last_response = response
                break

        if last_response is None:
            raise CodexAPIError("No terminal response received from the Codex backend.")
        return _parse_assistant_response(last_response), last_response

    def probe(self, *, model: str) -> None:
        self.complete_with_response(
            input_items=[_message_item("user", "Reply with the word pong.")],
            model=model,
            max_output_tokens=32,
            reasoning_effort="medium",
            text_verbosity="low",
        )

    @staticmethod
    def _to_api_error(response: requests.Response) -> CodexAPIError:
        status = response.status_code
        text = ""
        try:
            text = response.text
        except Exception:
            text = ""

        safe_excerpt = text[:1000]
        message = f"Codex backend request failed (HTTP {status})."
        code: str | None = None
        detail: str | None = None

        try:
            parsed = json.loads(text) if text else None
            if isinstance(parsed, dict):
                error = parsed.get("error")
                if isinstance(error, dict):
                    raw_code = error.get("code") or error.get("type")
                    code = raw_code if isinstance(raw_code, str) else None
                raw_detail = parsed.get("detail")
                detail = raw_detail if isinstance(raw_detail, str) else None
            if code:
                message = f"Codex backend request failed (HTTP {status}, {code})."
        except Exception:
            pass

        haystack = f"{code or ''} {detail or ''} {text}".lower()
        is_usage_limit = any(
            token in haystack
            for token in (
                "usage_limit_reached",
                "usage_not_included",
                "rate_limit_exceeded",
                "usage limit",
                "too many requests",
            )
        )
        if status == 404 and is_usage_limit:
            status = 429
            message = (
                "Codex usage limit reached for your ChatGPT subscription "
                "(treated as HTTP 429)."
            )
        if safe_excerpt:
            message = f"{message} Response excerpt: {safe_excerpt}"
        return CodexAPIError(message, status_code=status)


@dataclass
class OpenAIOAuthChat(Model):
    """Agno model that uses ChatGPT OAuth tokens against the Codex backend."""

    id: str = "gpt-5.3-codex"
    name: str = "OpenAIOAuthChat"
    provider: str = "OpenAI"
    supports_native_structured_outputs: bool = False
    supports_json_schema_outputs: bool = False

    reasoning_effort: Optional[str] = "medium"
    verbosity: Optional[Literal["low", "medium", "high"]] = "medium"
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    timeout: Optional[float] = 60.0
    max_backend_retries: int = 2
    include: Optional[list[str]] = None
    base_url: str = CODEX_BASE_URL
    system_prompt_mode: SystemPromptMode = "strict"

    auth_store: Optional[AuthStore] = None
    _client: Optional[_OAuthBackendClient] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self._client = _OAuthBackendClient(
            auth_store=self.auth_store,
            base_url=self.base_url,
            timeout_s=float(self.timeout or 60.0),
            max_retries=int(self.max_backend_retries),
        )

    def _get_client(self) -> _OAuthBackendClient:
        if self._client is None:
            self._client = _OAuthBackendClient(
                auth_store=self.auth_store,
                base_url=self.base_url,
                timeout_s=float(self.timeout or 60.0),
                max_retries=int(self.max_backend_retries),
            )
        return self._client

    def _build_model_response(self, response: dict[str, Any]) -> ModelResponse:
        parsed = _parse_assistant_response(response)
        return ModelResponse(
            role="assistant",
            content=parsed["content"],
            tool_calls=parsed["tool_calls"],
            provider_data=_extract_response_metadata(response),
            response_usage=_extract_usage_metrics(response),
        )

    def invoke(
        self,
        messages: list[Message],
        assistant_message: Message,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, dict[str, Any]]] = None,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        compress_tool_results: bool = False,
    ) -> ModelResponse:
        try:
            assistant_message.metrics.start_timer()
            system_texts = (
                _extract_system_texts(messages)
                if self.system_prompt_mode == "strict"
                else []
            )
            extra_instructions = (
                _build_extra_instructions(system_texts)
                if self.system_prompt_mode == "strict"
                else None
            )

            _, response = self._get_client().complete_with_response(
                input_items=_to_input_items(
                    messages,
                    system_prompt_mode=self.system_prompt_mode,
                    compress_tool_results=compress_tool_results,
                ),
                model=self.id,
                tools=tools,
                tool_choice=tool_choice,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                reasoning_effort=self.reasoning_effort,
                text_verbosity=self.verbosity,
                include=self.include,
                extra_instructions=extra_instructions,
                response_format=response_format,
            )
            assistant_message.metrics.stop_timer()
            return self._build_model_response(response)
        except NotAuthenticatedError as exc:
            raise ModelAuthenticationError(
                message=str(exc),
                model_name=self.name,
            ) from exc
        except CodexAPIError as exc:
            raise ModelProviderError(
                message=str(exc),
                status_code=exc.status_code,
                model_name=self.name,
                model_id=self.id,
            ) from exc
        except Exception as exc:
            raise ModelProviderError(
                message=str(exc),
                model_name=self.name,
                model_id=self.id,
            ) from exc

    async def ainvoke(
        self,
        messages: list[Message],
        assistant_message: Message,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, dict[str, Any]]] = None,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        compress_tool_results: bool = False,
    ) -> ModelResponse:
        return await asyncio.to_thread(
            self.invoke,
            messages,
            assistant_message,
            response_format,
            tools,
            tool_choice,
            run_response,
            compress_tool_results,
        )

    def invoke_stream(
        self,
        messages: list[Message],
        assistant_message: Message,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, dict[str, Any]]] = None,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        compress_tool_results: bool = False,
    ) -> Iterator[ModelResponse]:
        yield self.invoke(
            messages=messages,
            assistant_message=assistant_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
            compress_tool_results=compress_tool_results,
        )

    async def ainvoke_stream(
        self,
        messages: list[Message],
        assistant_message: Message,
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        tools: Optional[list[dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, dict[str, Any]]] = None,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        compress_tool_results: bool = False,
    ) -> AsyncIterator[ModelResponse]:
        yield await self.ainvoke(
            messages=messages,
            assistant_message=assistant_message,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            run_response=run_response,
            compress_tool_results=compress_tool_results,
        )

    def _parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        if isinstance(response, dict):
            return self._build_model_response(response)
        return ModelResponse(content=str(response) if response is not None else "")

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        if not isinstance(response, dict):
            return ModelResponse()
        event_type = str(response.get("type") or "")
        if event_type == "response.output_text.delta":
            delta = response.get("delta")
            return ModelResponse(content=delta if isinstance(delta, str) else "")
        if event_type == "response.output_item.done":
            item = response.get("item")
            if isinstance(item, dict) and item.get("type") == "function_call":
                call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
                name = item.get("name")
                arguments = item.get("arguments")
                if isinstance(name, str):
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments or {}, separators=(",", ":"))
                    return ModelResponse(
                        tool_calls=[
                            {
                                "id": str(call_id),
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                },
                            }
                        ]
                    )
        if _is_terminal_event(response):
            terminal = response.get("response")
            if isinstance(terminal, dict):
                return self._build_model_response(terminal)
        return ModelResponse()

    def probe(self) -> None:
        self._get_client().probe(model=self.id)
