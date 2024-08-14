# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean

# Install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Set the working directory
WORKDIR /app

# Copy the conda environment file
COPY env.yml .

# Create the conda environment
RUN micromamba create -f env.yml -y -n dis2

# Clean up
RUN micromamba clean --all --yes

# Set up the conda environment
ENV PATH=/opt/conda/envs/dis2/bin:$PATH
ENV CONDA_DEFAULT_ENV=dis2
ENV LMFE_MAX_CONSECUTIVE_WHITESPACES=2

# Ensure micromamba runs in the correct shell
SHELL ["micromamba", "run", "-n", "dis2", "/bin/bash", "-c"]

RUN pip install auto-gptq==0.7.1+cu118 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

COPY server/ /app/server/
# COPY models/ ./models/

# Expose the port the ML server runs on
EXPOSE 5000

# Command to run the ML server
CMD ["micromamba", "run", "-n", "dis2", "python", "/app/server/cuda.py"]
# RUN micromamba run -n dis2 python /app/server/cuda.py
