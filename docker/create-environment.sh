#!/bin/sh

# =============================================================================
# File        : create_environment.sh
# Description : This file creates a Docker's image named 'stnweb:v1.1'.
#               It uses the Dockerfile file. 
# =============================================================================

docker build . --tag stnweb:v1.1 
