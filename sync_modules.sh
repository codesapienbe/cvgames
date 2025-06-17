#! /bin/bash


for dir in */; do if [ -d "$dir" ] && [ -f "${dir}__init__.py" ]; then dir_name=${dir%/}; echo "$dir_name = \"$dir_name:main\"" >> pyproject.toml; fi; done