FROM ghcr.io/astral-sh/uv:python3.13-alpine

RUN addgroup -S sandbox && adduser -s /bin/sh -u 1000 -D -G sandbox sandbox
USER sandbox
WORKDIR /home/sandbox

RUN uv init
RUN uv add sympy numpy scipy

CMD ["python"]
