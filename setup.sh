#!/bin/bash
# Pre-deployment script to pull DVC-tracked model file(s)
echo "Running setup.sh: pulling DVC artifacts..."
dvc pull
