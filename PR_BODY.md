## Purpose

Introduce Structured Outputs for Azure OpenAI (Chat Completions) in the LLM post-processing path to:

- reduce model "free-form" additions,
- enforce a predictable JSON return format,
- simplify and stabilize client-side parsing of LLM results.

## Scope

- `src/llm_processor.py`
  - In `_process_azure_openai`, enable Structured Outputs via `response_format` with `json_schema` (supported on API versions that contain `preview` or start with `2025`).
  - Parse `message.content` as JSON and return the `processed_and_cleaned_transcript` field. If parsing fails, safely fall back to the previous plain text behavior.
  - Keep the standard `messages` payload (`system` + `user`) and continue to use Chat Completions (`.../chat/completions`).

- Optional SDK imports and startup resilience:
  - Relax imports for `ollama`, `groq`, and `google.generativeai` so absence of these SDKs does not crash module import; log and fall back instead.

- `src/transcription.py`
  - Optional imports for `OpenAI`/`Groq` with safe fallbacks (environment/test stability only; no logic change for Azure).

- Helper script: `migrate_azure_key.py` (required by tests) to move `azure_openai_api_key` into the system keyring.

## How it works (Azure LLM)

- For Azure API versions containing `preview` or starting with `2025`, the request body includes:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "processed_transcript_schema",
      "schema": {
        "type": "object",
        "properties": {
          "processed_and_cleaned_transcript": {"type": "string"}
        },
        "required": ["processed_and_cleaned_transcript"],
        "additionalProperties": false
      },
      "strict": true
    }
  }
}
```

- On response:
  - Attempt `json.loads(message.content)` and return `processed_and_cleaned_transcript`.
  - If that fails (e.g., older API or atypical response), fall back to `message.content`.

## Behavior changes

- For Azure LLM with API version `2025-*` or `*-preview`: model returns JSON, the app extracts only `processed_and_cleaned_transcript`, and that is what gets typed.
- For older API versions (e.g., `2024-02-01`): Structured Outputs are not requested; behavior remains unchanged.

## Configuration

- No changes are required other than optionally using a newer Azure API version (e.g., `2025-01-01-preview`) to enable Structured Outputs.
- Optional recommendation: set `temperature = 0` for maximal determinism in edit-only mode.

## Tests / verification

- Azure LLM path tests pass (`tests/test_azure_openai_llm.py`, selected integration in `tests/test_azure_openai_llm_integration.py`).
- Manual:
  1) Set `api_type = azure_openai` in LLM post-processing.
  2) Set `azure_openai_llm_api_version = 2025-01-01-preview` (or another version that supports Structured Outputs).
  3) Choose a compatible `gpt-4.1`/`4o` family deployment.
  4) Run a transcription; expect a clean output sourced from JSON only.

## Backward compatibility / risks

- Backward compatible: if the backend does not support `json_schema`, logic falls back to plain text (`message.content`).
- Defensive fallback and additional logging; no UI changes.

## Motivation

- Structured Outputs significantly reduce model verbosity and enforce a format that is simple to parse and stable for downstream processing.


