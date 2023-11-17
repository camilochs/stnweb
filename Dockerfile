# Use phusion/baseimage as base image. To make your builds
# reproducible, make sure you lock down to a specific version, not
# to `latest`! See
# https://github.com/phusion/baseimage-docker/blob/master/Changelog.md
# for a list of version numbers.
FROM phusion/baseimage:jammy-1.0.1

RUN apt-get update -y && \
    apt-get install --no-install-recommends -y \
        build-essential \
        r-base-dev \
        python3.11 python3-pip \
        r-base=4.1.2-1ubuntu2  \
        nodejs=12.22.9~dfsg-1ubuntu3.1 \
        npm=8.5.1~ds-1 && \
    apt-get clean all && \
    rm -rf /var/lib/apt/lists/*

# R dependencies
RUN R -e "install.packages(c('igraph', 'dply', 'tidyr', 'gtools'), dependencies=TRUE, repos='http://cran.rstudio.com/')" 

# Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.23.5 scipy==1.9.1 Werkzeug==2.2.2 Flask==2.2.2 Flask-Cors==3.0.10 treap==2.0.10 gunicorn==21.2.0 

RUN npm install http-server -g

COPY . /develop

#EXPOSE 5001
EXPOSE 8081

COPY docker/docker-entrypoint.sh /usr/bin/docker-entrypoint.sh
RUN ["chmod", "+x", "/usr/bin/docker-entrypoint.sh"]
ENTRYPOINT ["/usr/bin/docker-entrypoint.sh"]


RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

