#!/bin/bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit docker.io nvidia-utils-535 libvulkan1
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

sudo mkdir -p /opt/google/cuda-installer
cd /opt/google/cuda-installer/ || exit
sudo curl -fSsL -O https://github.com/GoogleCloudPlatform/compute-gpu-installation/releases/download/cuda-installer-v1.1.0/cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver
sudo python3 cuda_installer.pyz install_cuda
sudo python3 cuda_installer.pyz verify_cuda

# Wait for Docker to be ready
while ! docker info > /dev/null 2>&1; do
    echo "Waiting for Docker to start..."
    sleep 1
done

# Build and run the ML server container
sudo docker build -t ml-server:latest -f /tmp/cuda.dockerfile .
sudo systemctl restart docker
sudo docker run -d --gpus all -p 5000:5000 ml-server:latest

# Create a marker file to indicate the startup script has completed
sudo touch /var/log/startup-script-complete