FROM ghcr.io/astral-sh/uv:0.9.24@sha256:816fdce3387ed2142e37d2e56e1b1b97ccc1ea87731ba199dc8a25c04e4997c5 AS uv
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df AS runtime
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/src MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /opt/quest
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev
RUN useradd --create-home --uid 1000 researcher
USER researcher
WORKDIR /workspace
ENTRYPOINT ["quest"]

FROM runtime AS lab
USER root
WORKDIR /opt/quest
RUN uv sync --locked --no-dev --group lab --group data --group tracking
RUN chown -R researcher:researcher /opt/venv /opt/quest
COPY docker/jupyter_server_config.py /etc/jupyter/jupyter_server_config.py
USER researcher
WORKDIR /workspace
ENTRYPOINT ["jupyter", "lab"]
CMD ["--config=/etc/jupyter/jupyter_server_config.py"]
