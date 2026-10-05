"""Provider-specific initialization, configuration and prompts."""

from .artifactnet import ArtifactNetProvider
from .openspec import OpenSpecProvider

PROVIDERS = {"openspec": OpenSpecProvider, "openspec_artifactnet": ArtifactNetProvider}


def get_provider_class(name: str):
    if not isinstance(name, str) or name not in PROVIDERS:
        raise ValueError(f"Unsupported workflow provider: {name!r}")
    return PROVIDERS[name]


def create_provider(config, runtime):
    return get_provider_class(config["provider"])(config, runtime)
