#!/bin/bash

export R_LIBS="/usr/local/lib/R/site-library/"

# start api
cd /develop/api/
export FLASK_APP=stn.py
flask run --host 0.0.0.0 -p 5000 &

# start web
cd /develop/website/
http-server . -a 0.0.0.0 -p 8081 -c-1
