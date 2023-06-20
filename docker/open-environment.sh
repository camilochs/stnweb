#!/bin/sh

# =============================================================================
# File        : open_environment.sh
# Description : This file creates a Docker's container from the 
#               'algorithm' image and it offers a prompt inside the 
#               container for running bash commands.
# Notes       : Inside the container, all project files will be located under 
#               the '/develop' folder
# =============================================================================

image_name="stnweb:v1.1"

docker run -ti \
            -p 5000:5000 \
            -p 8081:8081 \
            --rm $image_name \
            /bin/bash
