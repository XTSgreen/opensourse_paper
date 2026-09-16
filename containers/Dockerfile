FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /opt/iot-reproducibility
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tutorials ./tutorials
RUN python -m pip install --no-cache-dir -e ".[test]"

CMD ["python", "scripts/reproduce.py", "smoke"]
