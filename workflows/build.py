"""Build workflow tools, including ArtifactNet from its own local Dockerfile."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["openspec", "openspec_artifactnet"], required=True)
    parser.add_argument("--node-image", default="node:22.21.1-bookworm-slim")
    parser.add_argument(
        "--artnet-source", type=Path, default=Path(__file__).resolve().parents[2] / "artnet",
        help="Local ArtifactNet checkout; builds its Dockerfile's runtime-base target",
    )
    parser.add_argument("--artnet-image", help="Tag for the CLI image built from --artnet-source")
    parser.add_argument("--openspec-version", default="1.7.0")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    target = "artifactnet" if args.provider == "openspec_artifactnet" else "openspec"
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.openspec_version):
        parser.error("--openspec-version must be an exact semantic version")
    artnet_image = None
    version = args.openspec_version
    if target == "artifactnet":
        source = args.artnet_source.expanduser().resolve()
        if not (source / "Dockerfile").is_file():
            parser.error(f"ArtifactNet Dockerfile not found in {source}; set --artnet-source")
        try:
            package = json.loads((source / "package.json").read_text())
        except (OSError, ValueError) as error:
            parser.error(f"Cannot read ArtifactNet package.json in {source}: {error}")
        version = package.get("version") if isinstance(package, dict) else None
        if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
            parser.error("ArtifactNet package.json must contain an exact semantic version")
        artnet_image = args.artnet_image or f"swe-workflow-artnet-cli:{version}"
        # Match Slop's dependency-first build; ArtifactNet owns pnpm, its lockfile,
        # compilation and production dependencies. Docker caches unchanged layers.
        # Latest aliases let the workflow YAML follow these local builds without
        # duplicating package.json's version; trials freeze the actual image ID.
        subprocess.run(
            [
                "docker", "build", "--target", "runtime-base", "-t", artnet_image,
                *(["-t", "swe-workflow-artnet-cli:latest"] if not args.artnet_image else []),
                str(source),
            ],
            check=True,
        )
    tag = args.tag or f"swe-workflow-{target}:{version}"
    with tempfile.TemporaryDirectory(prefix="swe-workflow-build-") as tmp:
        context = Path(tmp)
        dest = context / "workflows" / "docker"
        shutil.copytree(Path(__file__).parent / "docker", dest)
        subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(dest / "Dockerfile"),
                "--target",
                target,
                "-t",
                tag,
                *(["-t", "swe-workflow-artifactnet:latest"] if target == "artifactnet" and not args.tag else []),
                "--build-arg",
                f"NODE_IMAGE={args.node_image}",
                *(["--build-arg", f"ARTNET_IMAGE={artnet_image}"] if artnet_image else []),
                "--build-arg",
                f"OPENSPEC_VERSION={args.openspec_version}",
                str(context),
            ],
            check=True,
        )
    print(f"Built workflow tool image: {tag}")


if __name__ == "__main__":
    main()
