"""Construction of OpenAI SDK clients for the OpenAI API and for Azure/Foundry.

Azure's "v1" API (GA since August 2025) is OpenAI-compatible: pointing the plain
``OpenAI`` client at ``https://<resource>/openai/v1/`` removes the need for the
``AzureOpenAI`` class, for ``api-version`` query parameters, and for the separate
request-building code paths this app used to carry.

The legacy mode is kept because deployments predating the v1 rollout still answer
on ``/openai/deployments/<name>/...?api-version=<dated>``.
"""
from openai import AzureOpenAI, OpenAI

AZURE_MODE_V1 = 'v1'
AZURE_MODE_LEGACY = 'legacy'
AZURE_API_MODES = (AZURE_MODE_V1, AZURE_MODE_LEGACY)

# Hostnames Azure hands out for OpenAI-capable resources. All three serve the v1
# API; which one you get depends on how the resource was created (classic Azure
# OpenAI, AI Foundry project, or a multi-service Cognitive Services account).
AZURE_ENDPOINT_SUFFIXES = (
    '.openai.azure.com',
    '.services.ai.azure.com',
    '.cognitiveservices.azure.com',
)

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2


def normalize_azure_v1_base_url(endpoint: str) -> str:
    """Turn any Azure resource endpoint into a v1 base URL for the OpenAI client.

    Accepts a bare hostname, a full resource URL, or a URL that already ends in
    ``/openai/v1``, and tolerates a trailing slash on all of them.
    """
    if not endpoint:
        raise ValueError("Azure endpoint is not configured")

    base = str(endpoint).strip().rstrip('/')
    if not base.startswith(('http://', 'https://')):
        base = f"https://{base}"

    # Strip anything already pointing into the API so we can append it uniformly.
    for suffix in ('/openai/v1', '/openai'):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    return f"{base}/openai/v1/"


def build_openai_client(api_key: str, base_url: str = None, timeout: float = DEFAULT_TIMEOUT):
    """Client for the OpenAI API, or for any OpenAI-compatible endpoint."""
    if not api_key:
        raise ValueError("OpenAI API key is not configured")

    return OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=timeout,
        max_retries=DEFAULT_MAX_RETRIES,
    )


def build_azure_client(
    api_key: str,
    endpoint: str,
    api_mode: str = AZURE_MODE_V1,
    api_version: str = None,
    timeout: float = DEFAULT_TIMEOUT,
):
    """Client for Azure OpenAI / Microsoft Foundry.

    In v1 mode this is a plain ``OpenAI`` client aimed at the resource's
    ``/openai/v1/`` base URL; in legacy mode it is an ``AzureOpenAI`` client bound
    to a dated ``api-version``. In both modes the ``model`` argument of a request
    carries the *deployment* name, not the model name.
    """
    if not api_key:
        raise ValueError("Azure OpenAI API key is not configured")
    if not endpoint:
        raise ValueError("Azure OpenAI endpoint is not configured")

    if api_mode == AZURE_MODE_LEGACY:
        if not api_version:
            raise ValueError("Azure legacy mode requires an api_version")
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=str(endpoint).strip().rstrip('/'),
            api_version=api_version,
            timeout=timeout,
            max_retries=DEFAULT_MAX_RETRIES,
        )

    return OpenAI(
        api_key=api_key,
        base_url=normalize_azure_v1_base_url(endpoint),
        timeout=timeout,
        max_retries=DEFAULT_MAX_RETRIES,
    )


def resolve_azure_api_mode(configured_mode: str, api_version: str = None) -> str:
    """Pick the Azure API mode, inferring it from a legacy api-version if needed.

    A dated ``api-version`` (``2025-03-01-preview``) means the config predates the
    v1 rollout, so honour it rather than silently switching endpoints; the literal
    ``v1`` means the user already opted in.
    """
    mode = (configured_mode or '').strip().lower()
    if mode in AZURE_API_MODES:
        return mode

    version = (api_version or '').strip().lower()
    if version and version not in ('v1', '1', 'latest'):
        return AZURE_MODE_LEGACY
    return AZURE_MODE_V1
