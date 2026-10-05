# Workflow quick start

Run all commands from the repository root after completing the
[benchmark setup](../README.md) (Docker with Compose, dataset, and benchmark images).

Set your values in `.env_private`; the launcher loads it automatically:

```dotenv
SWE_MILESTONE_DATA_ROOT=/path/to/SWE-Milestone-data
UNIFIED_API_KEY=your-agent-api-key
# For a custom agent endpoint:
# UNIFIED_BASE_URL=https://your-api-endpoint/v1
# Required only for ArtifactNet:
OPENROUTER_API_KEY=your-openrouter-api-key
```

**OpenSpec**

```bash
uv run python -m workflows.build --provider openspec
uv run python scripts/run_all.py --config workflows/configs/trial_openspec.example.yaml
```

**OpenSpec + ArtifactNet**

Keep the local ArtifactNet source checkout at `../artnet`, or add
`--artnet-source /path/to/artnet` to the build command. Rebuild after updating it.

```bash
uv run python -m workflows.build --provider openspec_artifactnet
docker pull neo4j:5-community
uv run python scripts/run_all.py --config workflows/configs/trial_openspec_artifactnet.example.yaml
```

Once images are built, use just the `run_all.py` command to launch.

## Configuration

Edit the trial YAML passed to `--config`. You can also use your local
`trial_configs/openspec.yaml` or `trial_configs/artifactnet.yaml` instead of the
examples above.

| Trial field | What to configure |
| --- | --- |
| `data_root` | Dataset directory; the examples use `SWE_MILESTONE_DATA_ROOT` from `.env_private`. |
| `trial_name` | Experiment name. Leave off `_NNN` to let the launcher select the latest or next number. |
| `model`, `reasoning_effort` | Codex model and reasoning effort; keep `agent: codex`. |
| `repos` | Repositories to run. Both examples select Navidrome and scikit-learn. |
| `timeout` | Maximum seconds for one agent invocation, also subject to workflow budgets. |
| `workflow_config` | Workflow YAML path, relative to the trial YAML's directory. |

For a trial YAML under `trial_configs/`, set `workflow_config` to
`../workflows/configs/openspec.yaml` or `../workflows/configs/openspec_artifactnet.yaml`.

The workflow settings live in [configs/openspec.yaml](configs/openspec.yaml) and
[configs/openspec_artifactnet.yaml](configs/openspec_artifactnet.yaml):

| Workflow field | What to configure |
| --- | --- |
| `stages` | Ordered skill names and their prompt arguments. |
| `budgets.stage_seconds`, `budgets.milestone_seconds` | Total execution time allowed per stage and milestone, including retries. |
| `budgets.max_attempts` | Maximum attempts per stage, including the first attempt. |
| `runtime_image`, `versions` | Tool image and expected CLI versions; keep them consistent with your build. |
| `artifactnet.llm`, `artifactnet.embedding` | API URL, model and `api_key_env`; put the referenced keys in `.env_private`. |

Change workflow settings before creating a new experiment. Existing experiments
retain their original workflow configuration.

## Resume or rerun

After an interruption, rerun the same launch command without `--new` or `--force`:

```bash
uv run python scripts/run_all.py --config workflows/configs/trial_openspec.example.yaml
# For ArtifactNet:
uv run python scripts/run_all.py --config workflows/configs/trial_openspec_artifactnet.example.yaml
```

The launcher selects the latest `_NNN` experiment for that name. To resume a
specific older experiment, set `trial_name` to its full name, such as
`codex_openspec_001`, then run the command again.

- Keep the complete trial directory, its agent container and recorded tool
  images. For ArtifactNet, also keep the recorded Neo4j image and data volumes.
  If these were deleted, start a new experiment with `--new`.
- Completed workflow stages stay complete; unfinished stages continue within
  their remaining attempt and time budgets. Resuming does not reset those limits.
- Keep the workflow YAML unchanged when resuming. Changing it causes a configuration
  mismatch. Model, timeout, reasoning effort and milestone selection also retain
  their original values when resuming through this launcher.
- To start a separate experiment, append `--new` and use a `trial_name` without
  a numeric suffix. To erase and rerun the selected experiment, use `--force`;
  this deletes its existing trial data, including ArtifactNet's graph data.

## View progress and results

Use the full trial names printed by the launcher; replace `_001` below with the
actual suffix. The monitor shows status and benchmark scores, including after
the run finishes:

```bash
# Compare both workflows
uv run bash scripts/monitor.sh codex_openspec_001 codex_openspec_artifactnet_001

# Inspect individual milestones for one repository
uv run bash scripts/monitor.sh codex_openspec_001 --detail navidrome

# Refresh every 10 seconds
watch -n 10 'uv run bash scripts/monitor.sh codex_openspec_001 codex_openspec_artifactnet_001'
```

Each repository's files are under
`<data_root>/<repo>/e2e_trial/<full-trial-name>/`:

| Path within the trial | Contents |
| --- | --- |
| `evaluation/summary.json` | Milestone status and evaluation results. |
| `agent_stats.json` | Codex usage and cost statistics. |
| `log/agent_stdout.txt`, `log/agent_stderr.txt` | Agent execution logs. |
| `workflow/state.json` | Workflow stage status, attempts and errors. |
| `workflow/metrics.json` | Stage usage and ArtifactNet LLM/embedding usage by milestone and for the trial. |

`workflow/metrics.json` is written when the runner closes. ArtifactNet costs are
separate from the monitor's Codex costs; `cost_usd: null` means the cost is unknown.
For startup or infrastructure errors, check the launcher log at
`.swe-milestone/<repo>.log` in this repository.
