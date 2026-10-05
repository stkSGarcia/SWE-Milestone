FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 git iptables ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 -s /bin/sh fakeroot && mkdir /testbed && chown fakeroot:fakeroot /testbed
WORKDIR /testbed
CMD ["tail", "-f", "/dev/null"]
