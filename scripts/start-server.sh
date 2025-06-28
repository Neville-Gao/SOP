#!/usr/bin/env bash

set -e
set -x

uvicorn --host 0.0.0.0 --port 8080 --timeout-keep-alive 5 main:app