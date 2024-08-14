# Check Docker Socket File Permissions
sudo chown root:docker /var/run/docker.sock
sudo chmod 660 /var/run/docker.sock

# Ensure User is in Docker Group
sudo usermod -aG docker $USER
newgrp docker  # to apply the group change without logout

# Explicitly Set Docker Socket in Docker Commands
sudo docker -H unix:///var/run/docker.sock ps

# Restart Docker Service
sudo systemctl restart docker

# Run Docker Command with Sudo
sudo docker ps

# If necessary, reinstall Docker
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get install docker-ce docker-ce-cli containerd.io

# this one needs to work
docker ps