"""Python adapter that wraps OpenAI completions and exposes a SecRL SQL tool."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Dict, Optional, Sequence
from uuid import UUID

import pymysql
from pymysql.cursors import DictCursor

try:
    from litellm import acompletion as litellm_acompletion
    _LITELLM_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    litellm_acompletion = None
    _LITELLM_ERROR = exc

from atlas.connectors.openai import OpenAIAdapter
from atlas.connectors.registry import AdapterError, AgentAdapter
from atlas.connectors.utils import AdapterResponse, normalise_usage_payload
from atlas.config.models import LLMParameters, LLMProvider, OpenAIAdapterConfig


_SQL_CLIENT: "SecRLSqlClient | None" = None
_LLM_ADAPTER: AgentAdapter | None = None
_LLM_SIGNATURE: str | None = None


class _LiteLLMAdapter(AgentAdapter):
    """Lightweight adapter that proxies arbitrary LiteLLM providers."""

    def __init__(self, params: LLMParameters) -> None:
        if litellm_acompletion is None:
            raise AdapterError("litellm is required to call non-OpenAI providers") from _LITELLM_ERROR
        self._params = params

    async def ainvoke(self, prompt: str, metadata: Dict[str, Any] | None = None) -> AdapterResponse:
        messages = self._build_messages(prompt, metadata or {})
        kwargs = self._build_kwargs()
        kwargs["messages"] = messages
        try:
            response = await litellm_acompletion(**kwargs)
        except Exception as exc:  # pragma: no cover - network/runtime failures
            raise AdapterError("litellm request failed") from exc
        return self._parse_response(response)

    def _build_messages(self, prompt: str, metadata: Dict[str, Any]) -> list[Dict[str, Any]]:
        messages: list[Dict[str, Any]] = []
        entries = metadata.get("messages")
        if isinstance(entries, Sequence):
            for entry in entries:
                converted = self._convert_metadata_entry(entry)
                if converted:
                    messages.append(converted)
        elif metadata:
            messages.append({"role": "system", "content": json.dumps(metadata)})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _convert_metadata_entry(self, entry: Any) -> Dict[str, Any] | None:
        if not isinstance(entry, dict):
            return None
        role = entry.get("role") or self._map_entry_type(entry.get("type"))
        if not role:
            return None
        content = self._stringify_content(entry.get("content"))
        message: Dict[str, Any] = {"role": role, "content": content}
        tool_calls = entry.get("tool_calls")
        if role == "assistant" and tool_calls:
            message["tool_calls"] = self._normalise_tool_calls(tool_calls)
        if role == "tool" and entry.get("tool_call_id"):
            message["tool_call_id"] = entry["tool_call_id"]
        return message

    @staticmethod
    def _map_entry_type(entry_type: Any) -> str | None:
        mapping = {
            "system": "system",
            "human": "user",
            "ai": "assistant",
            "tool": "tool",
        }
        return mapping.get(str(entry_type or ""))

    @staticmethod
    def _stringify_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item) if isinstance(item, (dict, list)) else str(item))
            return "".join(parts)
        if isinstance(content, (dict, list)):
            return json.dumps(content)
        return str(content)

    @staticmethod
    def _normalise_tool_calls(raw_tool_calls: Any) -> list[Dict[str, Any]]:
        if raw_tool_calls is None:
            return []
        if isinstance(raw_tool_calls, str):
            try:
                raw_tool_calls = json.loads(raw_tool_calls)
            except json.JSONDecodeError:
                return []
        if isinstance(raw_tool_calls, dict):
            raw_tool_calls = [raw_tool_calls]
        cleaned: list[Dict[str, Any]] = []
        for call in raw_tool_calls:
            if isinstance(call, str):
                try:
                    call = json.loads(call)
                except json.JSONDecodeError:
                    continue
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            if not name:
                continue
            arguments = call.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            entry: Dict[str, Any] = {"name": name, "arguments": arguments}
            if call.get("id"):
                entry["id"] = call["id"]
            if call.get("type"):
                entry["type"] = call["type"]
            cleaned.append(entry)
        return cleaned

    def _build_kwargs(self) -> Dict[str, Any]:
        params = self._params
        api_key = os.getenv(params.api_key_env)
        if not api_key:
            raise AdapterError(f"environment variable '{params.api_key_env}' is not set")
        kwargs: Dict[str, Any] = {
            "model": params.model,
            "api_key": api_key,
            "temperature": params.temperature,
            "timeout": params.timeout_seconds,
        }
        if params.api_base:
            kwargs["api_base"] = params.api_base
        if params.max_output_tokens is not None:
            kwargs["max_tokens"] = params.max_output_tokens
        if params.top_p is not None:
            kwargs["top_p"] = params.top_p
        if params.additional_headers:
            kwargs["extra_headers"] = params.additional_headers
        if params.retry and params.retry.attempts > 1:
            # LiteLLM interprets max_retries as additional attempts.
            kwargs["max_retries"] = max(0, params.retry.attempts - 1)
        return kwargs

    def _parse_response(self, response: Any) -> AdapterResponse:
        try:
            choice = response["choices"][0]
            message = choice["message"]
            content = message.get("content")
            tool_calls_raw = message.get("tool_calls")
            tool_calls = self._normalise_tool_calls(tool_calls_raw) if tool_calls_raw is not None else None
            normalised_content = self._stringify_content(content) if content is not None else ""
            usage = normalise_usage_payload(response.get("usage"))
            return AdapterResponse(normalised_content, tool_calls=tool_calls, usage=usage)
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError("unexpected response format from litellm adapter") from exc


async def student_adapter(prompt: str, metadata: Dict[str, Any] | None = None) -> Any:
    """Entry point consumed by the Python adapter."""

    metadata = dict(metadata or {})
    tool_payload = metadata.get("tool")
    if tool_payload:
        metadata.pop("llm_config", None)
        return await _handle_tool_call(tool_payload)
    return await _invoke_llm(prompt, metadata)


async def _invoke_llm(prompt: str, metadata: Dict[str, Any]) -> Any:
    """Delegate language-model requests to the OpenAI adapter."""

    llm_config = metadata.pop("llm_config", None)
    adapter = _ensure_llm_adapter(llm_config)
    if metadata:
        response = await adapter.ainvoke(prompt, metadata=metadata)
    else:
        response = await adapter.ainvoke(prompt)
    return response


async def _handle_tool_call(payload: Dict[str, Any]) -> str:
    name = payload.get("name")
    if name == "secrl_sql":
        return await _execute_sql_tool(payload)
    if name in {"process_owner_lookup", "secrl_process_owner"}:
        return await _execute_process_owner_lookup(payload)
    raise AdapterError(f"Unsupported tool '{name}'")


async def _execute_sql_tool(payload: Dict[str, Any]) -> str:
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise AdapterError("Tool arguments must be an object")
    incident = arguments.get("incident_id") or arguments.get("incident")
    if incident is None:
        raise AdapterError("incident_id is required for SecRL SQL queries")
    sql = arguments.get("sql") or arguments.get("query") or arguments.get("statement")
    if not isinstance(sql, str) or not sql.strip():
        raise AdapterError("sql must be a non-empty string")
    params = arguments.get("params") or arguments.get("parameters") or {}
    if params and not isinstance(params, dict):
        raise AdapterError("params must be an object when provided")
    limit = arguments.get("limit") or arguments.get("max_rows") or 200
    try:
        limit_int = int(limit)
    except (TypeError, ValueError):
        raise AdapterError("limit must be an integer")
    if limit_int <= 0 or limit_int > 500:
        raise AdapterError("limit must be between 1 and 500")
    client = _ensure_sql_client()
    rows = await asyncio.to_thread(client.query, incident, sql, params, limit_int)
    payload = {
        "incident": client.database_name(incident),
        "row_count": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "rows": rows,
        "limit_applied": limit_int,
        "parameters": params,
        "sql": sql.strip(),
    }
    return json.dumps(payload, ensure_ascii=False)


async def _execute_process_owner_lookup(payload: Dict[str, Any]) -> str:
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise AdapterError("Tool arguments must be an object")
    incident = arguments.get("incident_id") or arguments.get("incident")
    if incident is None:
        raise AdapterError("incident_id is required for SecRL process owner lookup")
    host = arguments.get("host") or arguments.get("device") or arguments.get("device_name")
    if not isinstance(host, str) or not host.strip():
        raise AdapterError("host is required for SecRL process owner lookup")
    process_id = arguments.get("process_id") or arguments.get("pid")
    if process_id is None:
        raise AdapterError("process_id is required for SecRL process owner lookup")
    file_name = arguments.get("file_name") or arguments.get("process_name") or arguments.get("image")
    limit = arguments.get("limit") or 1
    try:
        limit_int = max(1, min(int(limit), 5))
    except (TypeError, ValueError):
        limit_int = 1
    client = _ensure_sql_client()
    rows, statement, query_params, account_names, account_summaries = await asyncio.to_thread(
        client.lookup_process_owner,
        incident,
        host,
        process_id,
        limit_int,
        file_name,
    )
    columns = list(rows[0].keys()) if rows else []
    parameter_snapshot = {
        "host": host,
        "process_id": process_id,
        "file_name": file_name,
    }
    serialised_parameters = {key: value for key, value in parameter_snapshot.items() if value is not None}
    instructions = None
    if account_names:
        formatted_accounts = ", ".join(account_names)
        instructions = (
            "Mandatory: quote the exact AccountName value(s) listed below in your final answer. "
            "Do not invent or transform these values."
        )

    payload = {
        "incident": client.database_name(incident),
        "row_count": len(rows),
        "columns": columns,
        "rows": rows,
        "limit_applied": limit_int,
        "parameters": serialised_parameters,
        "table": "DeviceProcessEvents",
        "sql": statement,
        "sql_parameters": query_params,
    }
    if account_names:
        payload["account_names"] = account_names
    if account_summaries:
        payload["account_summaries"] = account_summaries
    if instructions:
        payload["instructions"] = {
            "summary": instructions,
            "account_names": account_names,
        }
    json_blob = json.dumps(payload, ensure_ascii=False)
    if account_names:
        header_lines = [
            f"REQUIRED_ACCOUNT_NAME: {account_names[0]}",
            "ACCOUNT_SUMMARIES:",
            *account_summaries,
            "ACTION: Include the REQUIRED_ACCOUNT_NAME string verbatim in your final answer.",
            "RAW_PAYLOAD:",
        ]
        return "\n".join(header_lines) + "\n" + json_blob
    return json_blob


def _ensure_llm_adapter(llm_config: Dict[str, Any] | None) -> AgentAdapter:
    global _LLM_ADAPTER, _LLM_SIGNATURE
    if llm_config is None:
        raise AdapterError("llm_config missing from metadata")
    signature = json.dumps(llm_config, sort_keys=True)
    if _LLM_ADAPTER is not None and signature == _LLM_SIGNATURE:
        return _LLM_ADAPTER
    params = LLMParameters.model_validate(llm_config)
    if params.provider in {LLMProvider.OPENAI, LLMProvider.AZURE_OPENAI}:
        adapter = OpenAIAdapter(
            OpenAIAdapterConfig(
                name="arcops-cyber-student-llm",
                system_prompt="",
                llm=params,
            )
        )
    else:
        adapter = _LiteLLMAdapter(params)
    _LLM_ADAPTER = adapter
    _LLM_SIGNATURE = signature
    return adapter


def _ensure_sql_client() -> "SecRLSqlClient":
    global _SQL_CLIENT
    if _SQL_CLIENT is None:
        host = os.getenv("ATLAS_SECRL_HOST", "127.0.0.1")
        port = int(os.getenv("ATLAS_SECRL_PORT", "3307"))
        user = os.getenv("ATLAS_SECRL_USER", "atlas")
        password = os.getenv("ATLAS_SECRL_PASSWORD", "atlas")
        _SQL_CLIENT = SecRLSqlClient(host=host, port=port, user=user, password=password)
    return _SQL_CLIENT


@dataclass(slots=True)
class SecRLSqlClient:
    host: str
    port: int
    user: str
    password: str

    def database_name(self, incident: Any) -> str:
        incident_str = str(incident).strip()
        if incident_str.startswith("incident_"):
            db_name = incident_str
        else:
            db_name = f"incident_{incident_str}"
        if not db_name.replace("_", "").isalnum():
            raise AdapterError("Invalid incident identifier")
        return db_name

    def query(
        self,
        incident: Any,
        sql: str,
        params: Dict[str, Any],
        limit: int,
    ) -> list[dict[str, Any]]:
        db_name = self.database_name(incident)
        cleaned_sql = self._validate_sql(sql)
        serialised_params = {key: self._coerce_param(value) for key, value in params.items()}
        try:
            connection = pymysql.connect(
                host=self.host,
                port=int(self.port),
                user=self.user,
                password=self.password,
                database=db_name,
                cursorclass=DictCursor,
                autocommit=True,
                charset="utf8mb4",
            )
        except pymysql.MySQLError as exc:  # pragma: no cover - connection errors
            raise AdapterError(f"Failed to connect to SecRL database '{db_name}': {exc}") from exc
        try:
            with connection.cursor() as cursor:
                try:
                    cursor.execute(cleaned_sql, serialised_params or None)
                    rows = cursor.fetchmany(limit)
                except pymysql.MySQLError as exc:  # pragma: no cover - driver errors
                    raise AdapterError(f"SecRL SQL execution failed: {exc}") from exc
        finally:
            connection.close()
        return [self._serialise_row(row) for row in rows]

    def _validate_sql(self, sql: str) -> str:
        statement = sql.strip().rstrip(";")
        lowered = statement.lower()
        if not lowered.startswith(("select", "show", "describe", "explain")):
            raise AdapterError("Only read-only SELECT statements are permitted")
        forbidden = ("insert ", "update ", "delete ", "drop ", "alter ", "create ", "grant ", "revoke ", "truncate ", "replace ")
        if any(keyword in lowered for keyword in forbidden):
            raise AdapterError("Destructive or write operations are not allowed")
        return statement

    def _coerce_param(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)):
            return value
        if value is None:
            return None
        return str(value)

    def _serialise_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self._serialise_value(value) for key, value in row.items()}

    def _serialise_value(self, value: Any) -> Any:
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except Exception:
                return value.hex()
        return value

    def lookup_process_owner(
        self,
        incident: Any,
        host: str,
        process_id: Any,
        limit: int,
        file_name: Optional[str] = None,
    ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
        if limit <= 0:
            raise AdapterError("limit must be greater than zero")

        host_clean = host.strip()
        if not host_clean:
            raise AdapterError("host must be a non-empty string")

        host_short = host_clean.split(".", 1)[0]
        process_id_str = str(process_id).strip()
        if not process_id_str:
            raise AdapterError("process_id must be a non-empty value")

        where_clauses = [
            "(LOWER(DeviceName) LIKE %(host_like)s OR LOWER(DeviceName) = %(host_exact)s)",
            "CAST(ProcessId AS CHAR) = %(process_id)s",
        ]
        params: Dict[str, Any] = {
            "host_like": f"{host_short.lower()}.%",
            "host_exact": host_clean.lower(),
            "process_id": process_id_str,
            "row_limit": int(limit),
        }

        if file_name:
            params["file_name"] = file_name.strip().lower()
            where_clauses.append("LOWER(FileName) = %(file_name)s")

        statement = f"""
            SELECT
                TimeGenerated,
                DeviceName,
                ProcessId,
                FileName,
                FolderPath,
                ProcessCommandLine,
                AccountDomain,
                AccountName,
                AccountUpn,
                AccountSid,
                ReportId,
                InitiatingProcessFileName,
                InitiatingProcessCommandLine,
                InitiatingProcessAccountDomain,
                InitiatingProcessAccountName,
                InitiatingProcessAccountUpn,
                InitiatingProcessAccountSid
            FROM DeviceProcessEvents
            WHERE {" AND ".join(where_clauses)}
            ORDER BY TimeGenerated DESC
            LIMIT %(row_limit)s
        """

        rows = self.query(incident, statement, params, int(limit))
        serialised_params = {key: self._coerce_param(value) for key, value in params.items()}
        account_names: list[str] = []
        account_summaries: list[str] = []
        for row in rows:
            account = row.get("AccountName") or row.get("AccountUpn") or row.get("AccountSid")
            if isinstance(account, str):
                if account not in account_names:
                    account_names.append(account)
                account_display = account
            else:
                account_display = str(account) if account is not None else "UNKNOWN"
            device = row.get("DeviceName") or "UNKNOWN_DEVICE"
            pid = row.get("ProcessId") or process_id_str
            timestamp = row.get("TimeGenerated") or row.get("Timestamp")
            summary = f"Device={device}, ProcessId={pid}, AccountName={account_display}, TimeGenerated={timestamp}"
            account_summaries.append(summary)
        return rows, statement.strip(), serialised_params, account_names, account_summaries


__all__ = ["student_adapter"]
