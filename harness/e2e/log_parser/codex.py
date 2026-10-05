"""OpenAI Codex agent log parser implementation."""

import json
import logging
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.e2e.log_parser.base import AgentLogParser, register_parser
from harness.e2e.log_parser.models import NativeUsageUnit, ToolCallRecord
from harness.e2e.pricing import calculate_cost, has_tiered_pricing
from harness.e2e.pricing import resolve_pricing as _resolve_pricing_shared

logger = logging.getLogger(__name__)


@register_parser("codex")
class CodexLogParser(AgentLogParser):
    """Parser for OpenAI Codex logs.

    Codex outputs newline-delimited JSON events when run with --json flag.
    """

    FRAMEWORK_NAME = "codex"

    # Codex home directory in container
    CODEX_HOME = "/home/fakeroot/.codex"
    SESSIONS_DIR = f"{CODEX_HOME}/sessions"

    def extract_trace(self, container_name: str, output_dir: Path) -> bool:
        """Extract Codex execution trace.

        Codex doesn't have a dedicated trace extraction tool like claude-extract.
        We rely on the JSON output captured during execution.

        Args:
            container_name: Name of the Docker container
            output_dir: Directory to store trace files

        Returns:
            True if successful (always returns True as we use stdout logs)
        """
        logger.info("Codex trace extraction: using stdout JSON logs")
        # Codex traces are captured via --json output during execution
        # No additional extraction needed
        return True

    def extract_raw_logs(
        self,
        container_name: str,
        output_dir: Path,
        session_id: Optional[str] = None,
    ) -> Path:
        """Extract Codex logs from container.

        Copies the ~/.codex/sessions/ contents to output directory.
        Session files are stored as: sessions/{year}/{month}/{day}/rollout-{ts}-{thread_id}.jsonl

        Args:
            container_name: Docker container name
            output_dir: Directory to store extracted logs (typically log/)
            session_id: Optional thread_id to filter - only extract matching session file

        Returns:
            Path to extracted logs directory
        """
        # Store in {output_dir}/codex/ (agent name as directory, consistent with claude_code)
        logs_dir = output_dir / "codex"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Copy session JSONL files (flatten the date directory structure)
        try:
            # First, find all JSONL files in sessions directory
            find_result = subprocess.run(
                ["docker", "exec", container_name, "find", self.SESSIONS_DIR, "-name", "*.jsonl", "-type", "f"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if find_result.returncode == 0 and find_result.stdout.strip():
                jsonl_files = find_result.stdout.strip().split("\n")
                copied_count = 0

                for remote_path in jsonl_files:
                    if not remote_path:
                        continue

                    # Extract just the filename
                    filename = Path(remote_path).name

                    # Filter by session_id (thread_id) if provided
                    # Filename format: rollout-{timestamp}-{thread_id}.jsonl
                    if session_id:
                        if session_id not in filename:
                            continue

                    local_path = logs_dir / filename

                    # Copy each file
                    subprocess.run(
                        ["docker", "cp", f"{container_name}:{remote_path}", str(local_path)],
                        capture_output=True,
                        timeout=30,
                    )
                    copied_count += 1

                logger.info(f"Extracted {copied_count} Codex session files to {logs_dir}")
            else:
                logger.warning("No Codex session files found")

        except subprocess.TimeoutExpired:
            logger.warning("Timeout extracting Codex logs")
        except Exception as e:
            logger.warning(f"Error extracting Codex logs: {e}")

        return logs_dir

    def parse_tool_calls(self, log_dir: Path) -> List[ToolCallRecord]:
        """Parse tool calls from Codex JSON logs.

        Codex JSON events include tool/function calls in its output.

        Args:
            log_dir: Directory containing extracted logs

        Returns:
            List of tool call records sorted by timestamp
        """
        all_calls = []

        # Find all JSON/JSONL files
        json_files = list(log_dir.rglob("*.json")) + list(log_dir.rglob("*.jsonl"))

        for json_file in json_files:
            try:
                calls = self._parse_json_file(json_file)
                all_calls.extend(calls)
            except Exception as e:
                logger.warning(f"Error parsing {json_file}: {e}")

        # Sort by timestamp
        all_calls.sort(key=lambda x: x.timestamp if x.timestamp else datetime.min)

        logger.info(f"Parsed {len(all_calls)} tool calls from {len(json_files)} files")
        return all_calls

    def _parse_json_file(self, json_path: Path) -> List[ToolCallRecord]:
        """Parse a single JSON/JSONL file for tool calls.

        Args:
            json_path: Path to JSON file

        Returns:
            List of tool call records
        """
        calls = []

        with open(json_path, encoding="utf-8") as f:
            content = f.read().strip()

            # Try parsing as single JSON first
            try:
                data = json.loads(content)
                calls.extend(self._extract_tool_calls_from_event(data))
                return calls
            except json.JSONDecodeError:
                pass

            # Parse as JSONL
            for line_num, line in enumerate(content.split("\n"), 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                    calls.extend(self._extract_tool_calls_from_event(event))
                except json.JSONDecodeError as e:
                    logger.debug(f"Invalid JSON at {json_path}:{line_num}: {e}")

        return calls

    def _extract_tool_calls_from_event(
        self,
        event: Dict[str, Any],
    ) -> List[ToolCallRecord]:
        """Extract tool call records from a Codex JSON event.

        Codex events may contain:
        - "type": "function_call" or "tool_use"
        - Function/tool information in the event
        - Nested in "payload" for response_item events

        Args:
            event: Parsed JSON event

        Returns:
            List of tool call records
        """
        calls = []

        event_type = event.get("type", "")

        # Handle response_item events (Codex wraps function calls in payload)
        if event_type == "response_item" and "payload" in event:
            payload = event["payload"]
            payload_type = payload.get("type", "")

            if payload_type in ("function_call", "custom_tool_call"):
                # Add timestamp from outer event if not in payload
                if "timestamp" not in payload and "timestamp" in event:
                    payload["timestamp"] = event["timestamp"]
                call = self._create_tool_call_record(payload)
                if call:
                    calls.append(call)

            elif payload_type in ("function_call_output", "custom_tool_call_output"):
                # This is a result event - handled by parse_tool_results
                pass

        # Handle direct function calls (top-level)
        elif event_type in ("function_call", "tool_use", "tool_call"):
            call = self._create_tool_call_record(event)
            if call:
                calls.append(call)

        # Handle nested tool calls in messages
        if "tool_calls" in event:
            for tc in event.get("tool_calls", []):
                call = self._create_tool_call_record(tc)
                if call:
                    calls.append(call)

        # Handle Codex-specific command events
        if event_type == "command" or (event_type not in ("response_item",) and "command" in event):
            call = self._create_command_record(event)
            if call:
                calls.append(call)

        return calls

    def _create_tool_call_record(
        self,
        data: Dict[str, Any],
    ) -> Optional[ToolCallRecord]:
        """Create a ToolCallRecord from tool call data.

        Args:
            data: Tool call data

        Returns:
            ToolCallRecord or None if invalid
        """
        tool_id = data.get("id", data.get("call_id", ""))
        tool_name = data.get("name", data.get("function", {}).get("name", "unknown"))
        tool_input = data.get("input", data.get("arguments", data.get("function", {}).get("arguments", {})))

        # Parse input if it's a string
        if isinstance(tool_input, str):
            try:
                tool_input = json.loads(tool_input)
            except json.JSONDecodeError:
                tool_input = {"raw": tool_input}

        # Calculate input size
        input_size = len(json.dumps(tool_input, ensure_ascii=False).encode("utf-8"))

        # Parse timestamp
        timestamp = None
        timestamp_str = data.get("timestamp", data.get("created_at"))
        if timestamp_str:
            try:
                if isinstance(timestamp_str, (int, float)):
                    timestamp = datetime.fromtimestamp(timestamp_str)
                else:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    timestamp = timestamp.replace(tzinfo=None)
            except (ValueError, OSError):
                pass

        # Extract raw command for Bash-like tool calls (used by verification classifier)
        bash_command = None
        if tool_name in ("shell_command", "exec_command") and isinstance(tool_input, dict):
            bash_command = tool_input.get("command") or tool_input.get("cmd")

        return ToolCallRecord(
            id=tool_id,
            name=tool_name,
            timestamp=timestamp,
            success=not data.get("is_error", False),
            input_size=input_size,
            output_size=0,
            milestone_id=None,
            is_subagent=False,
            _bash_command=bash_command,
        )

    def _create_command_record(
        self,
        event: Dict[str, Any],
    ) -> Optional[ToolCallRecord]:
        """Create a ToolCallRecord from a command execution event.

        Args:
            event: Command event data

        Returns:
            ToolCallRecord or None
        """
        command = event.get("command", "")
        if isinstance(command, dict):
            command = command.get("command", "")

        if not command:
            return None

        timestamp = None
        timestamp_str = event.get("timestamp")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                timestamp = timestamp.replace(tzinfo=None)
            except ValueError:
                pass

        return ToolCallRecord(
            id=event.get("id", ""),
            name="shell_command",
            timestamp=timestamp,
            success=event.get("exit_code", 0) == 0,
            input_size=len(command.encode("utf-8")),
            output_size=len(event.get("output", "").encode("utf-8")),
            milestone_id=None,
            is_subagent=False,
            _bash_command=command if command else None,
        )

    # Pricing imported from harness.e2e.pricing (single source of truth).

    @classmethod
    def _resolve_pricing(
        cls, model: str, input_tokens: int, context_window: Optional[int] = None
    ) -> Dict[str, float]:
        """Resolve flat or tiered pricing for a model.

        For tiered models, the threshold is evaluated against the per-request
        input token count, including cached tokens.
        ``context_window`` remains accepted for compatibility, but model
        capacity does not determine the request's pricing tier.
        This prevents session-level aggregates
        (which can be millions of tokens) from incorrectly triggering the
        higher pricing tier. Aggregates without request receipts are handled
        separately as estimates.
        """
        rates = _resolve_pricing_shared(model, prompt_tokens=input_tokens)
        # Map canonical field name to Codex-expected field name
        return {
            "input": rates.get("input", 0),
            "cached_input": rates.get("cache_read", 0),
            "output": rates.get("output", 0),
        }

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        context_window: Optional[int] = None,
        cache_write_tokens: int = 0,
    ) -> float:
        """Calculate cost based on token usage.

        Cost formula:
            cost = input_cost + cached_cost + output_cost + cache_write_cost
        Where:
            - input_cost = (input_tokens - cached_tokens - cache_write_tokens) / 1M × input_price
            - cached_cost = cached_tokens / 1M × cached_input_price
            - output_cost = (output_tokens + reasoning_tokens) / 1M × output_price

        Note: For Codex models, output_tokens from turn.completed already includes
        reasoning tokens, so reasoning_tokens should typically be 0 to avoid
        double-counting.

        Args:
            model: Model name
            input_tokens: Total input tokens (including cached)
            output_tokens: Output tokens (already includes reasoning tokens for Codex)
            cached_tokens: Cached input tokens (subset of input_tokens)
            reasoning_tokens: Additional reasoning/thought tokens (default 0,
                only use if output_tokens does NOT include reasoning)
            context_window: Accepted for compatibility; ignored for pricing.
            cache_write_tokens: Cache writes included in total input tokens.

        Returns:
            Estimated cost in USD
        """
        return calculate_cost(
            model=model,
            # Non-cached input tokens
            input_tokens=max(0, input_tokens - cached_tokens - cache_write_tokens),
            # Output tokens (including reasoning tokens)
            output_tokens=output_tokens + reasoning_tokens,
            # Cached input tokens (discounted rate)
            cache_read_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            prompt_tokens=input_tokens,
        )

    def _read_rollout_usage(self, log_dir: Path) -> dict:
        """Read request receipts, deduplicated across copies of each thread."""
        threads = {}
        seen_totals = set()
        for jsonl_file in sorted(log_dir.glob("*.jsonl")):
            match = re.search(r"([0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12})$", jsonl_file.stem)
            thread_id = match.group(1) if match else jsonl_file.stem.removeprefix("rollout-")
            try:
                with jsonl_file.open(encoding="utf-8") as stream:
                    for line_num, line in enumerate(stream, 1):
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(event, dict):
                            continue
                        payload = event.get("payload", {})
                        if not isinstance(payload, dict):
                            continue
                        if event.get("type") == "session_meta" and isinstance(payload.get("id"), str):
                            thread_id = payload["id"]
                        record = threads.setdefault(
                            thread_id,
                            {
                                "model": "unknown",
                                "context_window": None,
                                "turns": 0,
                                "units": [],
                            },
                        )
                        if event.get("type") in {"session_meta", "turn_context"}:
                            model = payload.get("model")
                            if isinstance(model, str) and model:
                                record["model"] = model
                            if event["type"] == "turn_context":
                                # Count turn_context events as LLM API calls (turns)
                                # when native request receipts are unavailable.
                                record["turns"] += 1
                            continue
                        # Extract detailed token info from event_msg
                        if event.get("type") != "event_msg" or payload.get("type") != "token_count":
                            continue
                        info = payload.get("info")
                        if not isinstance(info, dict):
                            continue
                        # Get context window (metadata only; it does not select a price tier).
                        context_window = info.get("model_context_window")
                        if type(context_window) is int and context_window > 0:
                            record["context_window"] = context_window
                        last = info.get("last_token_usage")
                        if not isinstance(last, dict):
                            continue
                        fields = (
                            "input_tokens",
                            "cached_input_tokens",
                            "cache_write_input_tokens",
                            "output_tokens",
                            "reasoning_output_tokens",
                        )
                        try:
                            counts = tuple(max(0, int(last.get(key, 0) or 0)) for key in fields)
                            # Get total usage from last token_count event
                            cumulative = info.get("total_token_usage")
                            # Codex may emit duplicate token_count snapshots; dedupe by
                            # cumulative totals within each thread, including copied rollout files.
                            if isinstance(cumulative, dict) and cumulative:
                                key = (thread_id, tuple(int(cumulative.get(field, 0) or 0) for field in fields))
                            else:
                                key = (jsonl_file.name, line_num)
                        except (TypeError, ValueError):
                            continue
                        if not any(counts) or key in seen_totals:
                            continue
                        seen_totals.add(key)
                        input_tokens, cached_tokens, cache_write_tokens, output_tokens, reasoning_tokens = counts
                        timestamp = None
                        ts = event.get("timestamp")
                        if isinstance(ts, str):
                            try:
                                timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
                            except ValueError:
                                pass
                        # Note: output_tokens already includes reasoning tokens.
                        # reasoning_output_tokens is a detail, not additional output.
                        record["units"].append(
                            NativeUsageUnit(
                                id=f"{jsonl_file.name}:{line_num}",
                                source_type="turn",
                                timestamp=timestamp,
                                model=record["model"],
                                token_usage={
                                    "inputTokens": input_tokens,
                                    "outputTokens": output_tokens,
                                    "cacheReadInputTokens": cached_tokens,
                                    "cacheCreationInputTokens": cache_write_tokens,
                                    "reasoningOutputTokens": reasoning_tokens,
                                },
                                cost_usd=self._calculate_cost(
                                    record["model"],
                                    input_tokens,
                                    output_tokens,
                                    cached_tokens=cached_tokens,
                                    cache_write_tokens=cache_write_tokens,
                                ),
                            )
                        )
            except (OSError, UnicodeError) as error:
                logger.warning("Could not read Codex usage from %s: %s", jsonl_file, error)
        return threads

    def parse_native_usage_units(
        self,
        log_dir: Path,
        stdout_file: Path,
    ) -> List[NativeUsageUnit]:
        """Parse native turn-level usage units from Codex token_count events.

        Price each native request using its own input length.
        """
        return [unit for record in self._read_rollout_usage(log_dir).values() for unit in record["units"]]

    @staticmethod
    def _accumulate_usage(model_usage: dict, model: str, usage: dict, cost: float, source: str) -> None:
        entry = model_usage.setdefault(model, {})
        for field in (
            "inputTokens",
            "outputTokens",
            "cachedInputTokens",
            "cacheCreationInputTokens",
            "reasoningOutputTokens",
        ):
            entry[field] = entry.get(field, 0) + usage.get(field, 0)
        entry["costUSD"] = entry.get("costUSD", 0.0) + cost
        previous = entry.get("costSource", source)
        entry["costSource"] = source if previous == source else "mixed"
        entry["tokenAccounting"] = "openai"

    def parse_stdout_stats(
        self,
        stdout_file: Path,
        logs_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Parse agent_stdout.txt and JSONL files for accumulated statistics.

        Codex JSON output includes usage information in turn.completed events.
        JSONL files contain more detailed token_count events with context window info.

        Sum native request costs; mark stdout-only totals as estimates.
        turn.completed contains cumulative thread usage, not request lengths;
        use it only to recover usage not covered by native token_count receipts.

        Args:
            stdout_file: Path to agent_stdout.txt
            logs_dir: Optional path to logs directory containing JSONL files

        Returns:
            Dictionary with accumulated statistics
        """
        # Determine log directory: prefer logs_dir if provided, otherwise use stdout_file.parent/codex
        # Note: logs_dir from extract_raw_logs already points to the codex/ subdirectory
        log_dir = logs_dir if logs_dir is not None else stdout_file.parent / "codex"
        rollouts = self._read_rollout_usage(log_dir)
        # Track previous cumulative usage from turn.completed events.
        # Codex turn.completed reports thread-level cumulative totals that
        # increase monotonically across resume sessions. Keep the latest totals
        # per thread so earlier usage is not counted again after a resume.
        receipts = {}
        session_count = 0
        thread_ids = set()
        current_thread = "unidentified"
        # Parse stdout for session counts and final usage
        if stdout_file.exists():
            with stdout_file.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    # Count sessions from thread.started
                    if event.get("type") == "thread.started":
                        session_count += 1
                        nested = event.get("thread")
                        current_thread = str(
                            event.get("thread_id")
                            or (nested.get("id") if isinstance(nested, dict) else None)
                            or event.get("session_id")
                            or "unidentified"
                        )
                        thread_ids.add(current_thread)
                    # Extract usage from turn.completed events
                    # Note: total_turns is counted from native request receipts,
                    # falling back to turn_context events in JSONL files.
                    #
                    # IMPORTANT: turn.completed reports thread-level CUMULATIVE token
                    # totals. When a trial resumes (same thread_id), each session's
                    # turn.completed carries the running total so far. Keep the latest
                    # snapshot per thread, then subtract already counted request usage.
                    if event.get("type") != "turn.completed":
                        continue
                    usage = event.get("usage")
                    if not isinstance(usage, dict):
                        continue
                    try:
                        # Codex uses input_tokens/output_tokens (not prompt_tokens/completion_tokens)
                        # Update cumulative tracker
                        # Keep the latest totals per logical thread across resumes.
                        receipts[current_thread] = {
                            "inputTokens": max(0, int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)),
                            "outputTokens": max(
                                0, int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
                            ),
                            "cachedInputTokens": max(0, int(usage.get("cached_input_tokens", 0) or 0)),
                            "cacheCreationInputTokens": max(0, int(usage.get("cache_write_input_tokens", 0) or 0)),
                            "reasoningOutputTokens": max(0, int(usage.get("reasoning_output_tokens", 0) or 0)),
                        }
                    except (TypeError, ValueError):
                        continue
        if "unidentified" in receipts and len(rollouts) == 1:
            receipts[next(iter(rollouts))] = receipts.pop("unidentified")
        known_models = (
            {r["model"] for r in rollouts.values()} | {unit.model for r in rollouts.values() for unit in r["units"]}
        ) - {"unknown"}
        fallback_model = next(iter(known_models)) if len(known_models) == 1 else "unknown"
        model_usage = {}
        total_turns = 0
        for thread_id in sorted(rollouts.keys() | receipts.keys()):
            record = rollouts.get(thread_id, {})
            covered = {}
            units = record.get("units", [])
            total_turns += len(units) if units else record.get("turns", 0)
            for unit in units:
                usage = dict(unit.token_usage)
                usage["cachedInputTokens"] = usage.pop("cacheReadInputTokens", 0)
                self._accumulate_usage(model_usage, unit.model, usage, unit.cost_usd, "request_usage")
                for field, value in usage.items():
                    covered[field] = covered.get(field, 0) + value
            receipt = receipts.get(thread_id, {})
            # Compute the delta not covered by native request receipts.
            residual = {field: max(0, count - covered.get(field, 0)) for field, count in receipt.items()}
            if any(
                residual.get(field, 0)
                for field in ("inputTokens", "outputTokens", "cachedInputTokens", "cacheCreationInputTokens")
            ):
                model = record.get("model", fallback_model)
                if model == "unknown":
                    model = fallback_model
                # The aggregate cannot reveal how many requests crossed a tier.
                # Retain the legacy base-rate estimate, explicitly labeled.
                # Calculate cost per model using delta (with cached tokens)
                # Note: output_tokens already includes reasoning tokens.
                cost = calculate_cost(
                    model=model,
                    input_tokens=max(
                        0,
                        residual["inputTokens"] - residual["cachedInputTokens"] - residual["cacheCreationInputTokens"],
                    ),
                    output_tokens=residual["outputTokens"],
                    cache_read_tokens=residual["cachedInputTokens"],
                    cache_write_tokens=residual["cacheCreationInputTokens"],
                )
                self._accumulate_usage(model_usage, model, residual, cost, "aggregate_estimate")
                if has_tiered_pricing(model):
                    logger.warning(
                        "Codex thread %s lacks request-level usage for some tokens; "
                        "costSource=aggregate_estimate uses base-tier rates",
                        thread_id,
                    )
            # Add context window to model usage. Reasoning details were accumulated above.
            context_window = record.get("context_window")
            if context_window:
                for model in {unit.model for unit in units} | {record.get("model", "unknown")}:
                    if model in model_usage:
                        model_usage[model]["contextWindow"] = max(
                            model_usage[model].get("contextWindow", 0),
                            context_window,
                        )
        return {
            "total_cost_usd": sum(usage["costUSD"] for usage in model_usage.values()),
            "total_turns": total_turns,
            "modelUsage": model_usage,
            "session_count": session_count,
            "unique_session_count": len(thread_ids) if thread_ids else session_count,
        }

    def _normalize_native_usage_costs(self, native_usage_units: list[NativeUsageUnit], total_cost: float) -> None:
        """Keep per-request estimates unchanged.

        Any stdout-only residual is an aggregate estimate. Scaling known
        request costs to absorb that residual would erase their actual tiers.
        """

    def parse_tool_results(
        self,
        log_dir: Path,
        tool_calls: List[ToolCallRecord],
    ) -> None:
        """Update tool calls with result information from Codex logs.

        Parses result events from JSONL files and updates corresponding
        tool call records with success status and output size.

        Modifies tool_calls in place.

        Args:
            log_dir: Directory containing extracted logs
            tool_calls: List of tool call records to update
        """
        # Build lookup map by tool call ID
        calls_by_id = {tc.id: tc for tc in tool_calls if tc.id}

        if not calls_by_id:
            return

        # Find all JSON/JSONL files
        json_files = list(log_dir.rglob("*.json")) + list(log_dir.rglob("*.jsonl"))

        for json_file in json_files:
            try:
                with open(json_file, encoding="utf-8") as f:
                    content = f.read().strip()

                # Try parsing as single JSON first
                try:
                    data = json.loads(content)
                    self._update_tool_results_from_event(data, calls_by_id)
                    continue
                except json.JSONDecodeError:
                    pass

                # Parse as JSONL
                for line in content.split("\n"):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                        self._update_tool_results_from_event(event, calls_by_id)
                    except json.JSONDecodeError:
                        continue

            except Exception as e:
                logger.debug(f"Error parsing tool results from {json_file}: {e}")

    def _update_tool_results_from_event(
        self,
        event: Dict[str, Any],
        calls_by_id: Dict[str, ToolCallRecord],
    ) -> None:
        """Update tool call records from a result event.

        Args:
            event: Parsed JSON event
            calls_by_id: Mapping of tool call ID to ToolCallRecord
        """
        event_type = event.get("type", "")

        # Handle response_item with function_call_output payload (Codex format)
        if event_type == "response_item" and "payload" in event:
            payload = event["payload"]
            payload_type = payload.get("type", "")

            if payload_type in ("function_call_output", "custom_tool_call_output"):
                call_id = payload.get("call_id", "")
                if call_id and call_id in calls_by_id:
                    tc = calls_by_id[call_id]
                    output = payload.get("output", "")

                    # Check for error based on output content
                    if payload_type == "custom_tool_call_output":
                        # custom_tool_call_output uses "status" field
                        tc.success = payload.get("status") != "error"
                    else:
                        tc.success = "Exit code: 0" in output or not output.startswith("Error")

                    if isinstance(output, str):
                        tc.output_size = len(output.encode("utf-8"))
                    elif isinstance(output, (dict, list)):
                        tc.output_size = len(json.dumps(output, ensure_ascii=False).encode("utf-8"))
                return

        # Handle function/tool results (direct format)
        if event_type in ("function_call_result", "tool_result", "tool_call_result"):
            call_id = event.get("call_id", event.get("tool_use_id", event.get("id", "")))
            if call_id and call_id in calls_by_id:
                tc = calls_by_id[call_id]
                tc.success = not event.get("is_error", False)

                # Calculate output size
                output = event.get("output", event.get("content", event.get("result", "")))
                if isinstance(output, str):
                    tc.output_size = len(output.encode("utf-8"))
                elif isinstance(output, (dict, list)):
                    tc.output_size = len(json.dumps(output, ensure_ascii=False).encode("utf-8"))

        # Handle nested tool_results in messages
        if "tool_results" in event:
            for result in event.get("tool_results", []):
                call_id = result.get("call_id", result.get("id", ""))
                if call_id and call_id in calls_by_id:
                    tc = calls_by_id[call_id]
                    tc.success = not result.get("is_error", False)

                    output = result.get("output", result.get("content", ""))
                    if isinstance(output, str):
                        tc.output_size = len(output.encode("utf-8"))
                    elif isinstance(output, (dict, list)):
                        tc.output_size = len(json.dumps(output, ensure_ascii=False).encode("utf-8"))
