#!/bin/bash
python -m black . --exclude 'venv|notebooks'
docformatter -i -r . --exclude venv
isort .
