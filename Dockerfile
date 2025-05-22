ARG FROM_IMAGE=gitlab-registry.cern.ch/batch-team/dask-lxplus/lxdask-al9:latest
FROM ${FROM_IMAGE}

ARG CLUSTER=lxplus-el9

ADD . .

RUN echo "=======================================" && \
    echo "Installing HiggsDNA" && \
    echo "on cluster environment: $CLUSTER" && \
    echo "Current time:" $(date) && \
    echo "=======================================" && \
    yum -y update && \
    yum -y install git-lfs && \
    if [[ ${CLUSTER} == "lxplus-cc7" ]]; then \
        echo "Fixing dependencies in the image" && \
        conda install -y numba>=0.57.0 llvmlite==0.40.0 numpy>=1.22.0 && \
        python -m pip install -U dask-lxplus==0.3.2 dask-jobqueue==0.8.2; \
    fi && \
    installed_version=$(pip show pyarrow | grep 'Version:' | awk '{print $2}'); \
    if [ "$(printf '%s\n' "$installed_version" "11.0.0" | sort -V | head -n1)" = "11.0.0" ]; then \
        pip install --upgrade pyarrow; \
    fi && \
    echo "Installing HiggsDNA" && \
    python -m pip install . --verbose
