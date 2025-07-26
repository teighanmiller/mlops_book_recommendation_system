#!/bin/bash
export PYTHONPATH=.:src
./.venv/bin/pylint -rn -sn --recursive=y src tests dags
