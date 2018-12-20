#!/bin/bash

rm -rf dist/*

python3 setup.py sdist upload

python3 setup.py bdist_wheel upload

twine upload dist/*