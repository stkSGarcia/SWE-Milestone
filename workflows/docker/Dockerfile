# Build ahead of a trial; the benchmark's base image stays unchanged.
# Match Slop-Code-Bench's Node version while keeping the tool bundle relocatable.
ARG NODE_IMAGE=node:22.21.1-bookworm-slim
# build.py overrides this with the version read from ArtifactNet's package.json.
ARG ARTNET_IMAGE=swe-workflow-artnet-cli:latest
FROM ${NODE_IMAGE} AS openspec
ARG OPENSPEC_VERSION=1.7.0
RUN mkdir -p /opt/workflow/bin && \
    cp /usr/local/bin/node /opt/workflow/bin/node && \
    npm install -g --prefix /opt/workflow/npm @fission-ai/openspec@${OPENSPEC_VERSION} && \
    ln -s ../npm/bin/openspec /opt/workflow/bin/openspec
ENV PATH="/opt/workflow/bin:${PATH}"

# build.py first builds ArtifactNet's own Dockerfile with --target runtime-base.
# Keep this stage after openspec so OpenSpec alone needs no ArtifactNet image.
FROM ${ARTNET_IMAGE} AS artnet-source
FROM openspec AS artifactnet
COPY --from=artnet-source /app /opt/workflow/artnet
RUN ln -s ../artnet/bin/artnet.js /opt/workflow/bin/artnet
