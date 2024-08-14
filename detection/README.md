sudo apt update
sudo apt-get install jq
snap install google-cloud-cli --classic
gcloud init
sudo usermod -aG docker $USER
newgrp docker
docker -H unix:///var/run/docker.sock run --runtime=nvidia --rm nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi
docker stop $(docker ps -q)
docker rm $(docker ps -a -q)

docker system prune -a

# For local deployment
./deploy_local.sh config/service_account.json

docker exec -it python-server-api-dev /bin/bash
docker logs python-server-api-dev

curl -X GET http://localhost:5000/health
curl -X GET http://localhost:8080/health

curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"input":"Trump doesnt believe in Covid-19."}'
docker exec -it ml-server-dev /bin/bash
docker logs ml-server-dev
micromamba run -n dis2 python /app/server/cuda.py

python server/test_server.py

## locally test the container registry
docker build -t gcr.io/fine-sublime-429011-q2/disinformation-detection:latest -f run.dockerfile .
docker push gcr.io/fine-sublime-429011-q2/disinformation-detection:latest

docker run -d --name disinformation-detection \
  -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT=fine-sublime-429011-q2 \
  -e GOOGLE_CLOUD_ZONE=asia-southeast1-c \
  -e GCE_INSTANCE_NAME=ml-instance \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/service_account.json \
  -v $(pwd)/config/service_account.json:/app/service_account.json \
  gcr.io/fine-sublime-429011-q2/disinformation-detection:latest

# For production deployment
./deploy_prodv3.sh config/service_account.json --recreate-run

## testing the container
docker build -t gcr.io/fine-sublime-429011-q2/disinformation-detection:latest -f run.dockerfile .
docker push gcr.io/fine-sublime-429011-q2/disinformation-detection:latest

gcloud run deploy disinformation-detection     --image gcr.io/fine-sublime-429011-q2/disinformation-detection:latest     --platform managed     --region asia-southeast1     --set-env-vars GOOGLE_CLOUD_PROJECT=fine-sublime-429011-q2,GOOGLE_CLOUD_ZONE=asia-southeast1-c,GCE_INSTANCE_NAME=ml-instance     --allow-unauthenticated


### commands to test each section
curl -X GET https://python-server-api-gcr-jaokpic3ha-as.a.run.app/health

curl -X GET https://$(gcloud compute instances describe ml-instance --zone=asia-southeast1-c '--format=get(networkInterfaces[0].accessConfigs[0].natIP)'):5000/health

curl -X GET https://34.142.204.3:5000/health
curl -X POST https://34.142.204.3:5000/predict -H "Content-Type: application/json" -d '{"input":"Trump doesnt believe in Covid-19."}'

curl -X POST https://python-server-api-gcr-jaokpic3ha-as.a.run.app/predict -H "Content-Type: application/json" -d '{"input":"Trump doesnt believe in Covid-19."}'
<!-- https://python-server-api-gcr-jaokpic3ha-as.a.run.app -->

# For shutdown
./shutdown.sh config/service_account.json python-server-api-gcr

LMFE_MAX_CONSECUTIVE_WHITESPACES=2 python llm_handler.py
DEV=true python server.py


# to get the keys and cert
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout nginx-selfsigned.key -out nginx-selfsigned.crt -config openssl.cnf
sudo chown reisub-laptop nginx-selfsigned.*

# logging and debugging
gcloud projects get-iam-policy fine-sublime-429011-q2
gcloud container images list --repository=gcr.io/fine-sublime-429011-q2
gcloud run services describe disinformation-detection --region asia-southeast1

# make sure the firewall is setup properly
gcloud compute firewall-rules create allow-port-5000 \
    --network default \
    --direction INGRESS \
    --priority 1000 \
    --action ALLOW \
    --rules tcp:5000 \
    --source-ranges 0.0.0.0/0