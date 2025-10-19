"""Python adapter that wraps OpenAI completions and exposes a SecRL SQL tool."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Dict, Optional
from uuid import UUID

import pymysql
from pymysql.cursors import DictCursor

from atlas.connectors.openai import OpenAIAdapter
from atlas.connectors.registry import AdapterError
from atlas.config.models import LLMParameters, OpenAIAdapterConfig


_SQL_CLIENT: "SecRLSqlClient | None" = None
_OPENAI_ADAPTER: OpenAIAdapter | None = None
_OPENAI_SIGNATURE: str | None = None


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
    adapter = _ensure_openai_adapter(llm_config)
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


def _ensure_openai_adapter(llm_config: Dict[str, Any] | None) -> OpenAIAdapter:
    global _OPENAI_ADAPTER, _OPENAI_SIGNATURE
    if llm_config is None:
        raise AdapterError("llm_config missing from metadata")
    signature = json.dumps(llm_config, sort_keys=True)
    if _OPENAI_ADAPTER is not None and signature == _OPENAI_SIGNATURE:
        return _OPENAI_ADAPTER
    params = LLMParameters.model_validate(llm_config)
    adapter_config = OpenAIAdapterConfig(
        name="arcops-cyber-student-llm",
        system_prompt="",
        llm=params,
    )
    _OPENAI_ADAPTER = OpenAIAdapter(adapter_config)
    _OPENAI_SIGNATURE = signature
    return _OPENAI_ADAPTER


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
