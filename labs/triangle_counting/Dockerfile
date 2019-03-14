FROM c3sr/pangolin:amd64-cuda100-ece508-fc6066b as builder

LABEL maintainer="pearson@illinois.edu"

RUN apt-get update && apt-get install -y --no-install-suggests --no-install-recommends \
    curl \
    gzip \
&& rm -rf /var/lib/apt/lists/*

# add some graph data
ENV GRAPH_DIR=/graphs
ENV CMAKE_INSTALL_DIR=/opt/cmake
RUN mkdir /graphs
WORKDIR /graphs

# a synthetic graph
RUN curl -SLO https://graphchallenge.s3.amazonaws.com/synthetic/graph500-scale18-ef16/graph500-scale18-ef16_adj.tsv.gz
RUN gunzip graph500-scale18-ef16_adj.tsv.gz
RUN pangolin-tsv-to-bel.py graph500-scale18-ef16_adj.tsv

# SNAP graphs
RUN curl -SLO https://graphchallenge.s3.amazonaws.com/snap/amazon0302/amazon0302_adj.tsv
RUN pangolin-tsv-to-bel.py amazon0302_adj.tsv
RUN curl -SLO https://graphchallenge.s3.amazonaws.com/snap/roadNet-CA/roadNet-CA_adj.tsv
RUN pangolin-tsv-to-bel.py roadNet-CA_adj.tsv
# add cmake
RUN mkdir -p $CMAKE_INSTALL_DIR
RUN curl -SL https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4-Linux-x86_64.tar.gz | tar -xz --strip-components=1 -C $CMAKE_INSTALL_DIR
ENV PATH "${CMAKE_INSTALL_DIR}/bin:${PATH}"

# run cmake to precompile dependencies to speed up student builds
COPY . /src
WORKDIR /build
RUN cmake /src -DGRAPH_PREFIX_PATH=junk

# start over, but keep graphs, cmake, and hunter directory
FROM c3sr/pangolin:amd64-cuda100-ece508-fc6066b
ENV GRAPH_DIR=/graphs
ENV CMAKE_INSTALL_DIR=/opt/cmake
COPY --from=builder $GRAPH_DIR/*.bel $GRAPH_DIR/
COPY --from=builder $CMAKE_INSTALL_DIR $CMAKE_INSTALL_DIR
COPY --from=builder /root/.hunter /root/.hunter
# rai doesn't seem to pick this path up
ENV PATH "${CMAKE_INSTALL_DIR}/bin:${PATH}"
