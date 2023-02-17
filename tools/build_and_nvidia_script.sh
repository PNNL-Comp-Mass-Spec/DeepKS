DEEPKS_VERSION='0.0.11.5'
apptainer build --sandbox deepks-latest.sif Docker://benndrucker/deepks:${DEEPKS_VERSION} && \
cd deepks-latest.sif && \
cp /usr/bin/nvidia-smi ./usr/bin/ && \
cp /usr/bin/nvidia-debugdump ./usr/bin/ && \
cp /usr/bin/nvidia-cuda-mps-control ./usr/bin/ && \
cp /usr/bin/nvidia-cuda-mps-server ./usr/bin/ && \
cp /var/run/nvidia-persistenced/socket ./var/run/nvidia-persistenced/ && \
echo "Success!!!"