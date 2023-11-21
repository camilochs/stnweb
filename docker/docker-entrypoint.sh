#!/bin/bash

export R_LIBS="/usr/local/lib/R/site-library/"

# start api


cd /develop/website/
http-server . -a 0.0.0.0 -p 8081 -c-1 &

# start web
cd /develop/api/
mkdir logs
touch logs/error.log
touch logs/access.log

gunicorn --workers=8 --threads=3  --timeout 500 \
    --preload --reload --log-level debug --access-logfile logs/access.log --error-logfile logs/error.log \
     -b '0.0.0.0:5001' 'stn:app' 

