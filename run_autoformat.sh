#!/bin/bash
python -m black . --extend-exclude 'venv|notebooks'
docformatter -i -r . --exclude venv
isort . --skip logs --skip venv
