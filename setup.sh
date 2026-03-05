#!/bin/bash

# install uv if not exist
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed."
fi

# install dependencies
uv sync --no-dev
