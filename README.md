# Usage
Start by CD into detection. This is where you should be for all operations unless it is building the front end for the extension
`cd detection`

Add your hugging face key to ./detection/server/cuda.py on line 44

Create a google cloud account, create a project, create a service account, add a key that is json and put in detection/config folder
Start Logging API
Run `snap install google-cloud-cli --classic` to install gcloud locally, and then use `gcloud init` to select your project

For testing and debugging, it is helpful to have anaconda installed and running. Here is an example for how you could do this with mamba  
```
curl -o Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda install mamba -n base -c conda-forge
mamba env create -n dis2 --file 'env.yml'
mamba activate dis2
```


Further, run these commands to set up the rest of the system

```
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit docker.io nvidia-utils-535 libvulkan1
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker
```

## Local non-docker
Run these in seperate terminals, and then make curl requests to the proper URL (see "For local testing and debugging" section)
```
DEV=true GOOGLE_APPLICATION_CREDENTIALS=detection/service_account.json python server/run.py
DEV=true GOOGLE_APPLICATION_CREDENTIALS=detection/service_account.json python server/cuda.py
```

## Local docker
```
./deploy_local.sh config/service_account.json
```

To stop it locally, run
```
docker stop $(docker ps -q)
docker rm $(docker ps -a -q)
```

To completely wipe the docker contents from your system and free up space, run:
```
docker system prune -a
```


## Production
Start Cloud Run API
Start Compute Engine API
Request access to a GPU, and select server and such. I chose the n1-standard-4 server in zone asia-southeast1-c. These can be changed in the deploy_prodv3.sh file manually to match what you have chosen.

To deploy it on a server, run
`./deploy_prodv3.sh config/service_account.json --recreate-run --recreate-ml`

# For local testing and debugging
Starts two docker containers, one for the main bulk of the processing, and one as a front end and validation.

Port 8080 is the validation port (run.py), port 5000 is the LLM port (cuda.py)
```
./deploy_local.sh config/service_account.json
```

### Test that each are alive
curl -X GET http://localhost:8080/health
curl -X GET http://localhost:5000/health

### Test the full LLM Pipeline
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"input":"Joe Biden was the 46th president of the United States"}'

### Test the connection between the servers
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{"input":"Joe Biden was the 46th president of the United States"}'

## Debugging
### Validation Container
Prints out logs for validation container (run.py)
```
docker logs python-server-api-dev
```

Makes the validation container interactive
```
docker exec -it python-server-api-dev /bin/bash
```

If you need to run the script to test it in the container again, use the following. It also works outside of the docker container (if you installed micromamba, or it would just be mamba run, not micromamba run)
```
micromamba run -n dis2 python /app/server/run.py
```

### Pipeline Container
Prints out logs for pipeline container (cuda.py)
```
docker logs ml-server-dev
```

Makes the validation pipeline interactive
```
docker exec -it ml-server-dev /bin/bash
```

If you need to run the script to test it in the server again, use the following. It also works outside of the docker container (if you installed micromamba, or it would just be mamba run, not micromamba run)
```
micromamba run -n dis2 python /app/server/cuda.py
```

### Small test suite
Small test suite to check if everything works. All should be 200 except last, which should be 400
```
python server/test_server.py
```

## Manually building the containers
To build the containers by hand, go through the commands within. Your server will have a different $PROJECT_ID. It is your gcloud project id
```
docker build -t gcr.io/$PROJECT_ID/disinformation-detection:latest -f run.dockerfile .
docker push gcr.io/$PROJECT_ID/disinformation-detection:latest

docker run -d --name disinformation-detection \
  -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT=$PROJECT_ID \
  -e GOOGLE_CLOUD_ZONE=asia-southeast1-c \
  -e GCE_INSTANCE_NAME=ml-instance \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/service_account.json \
  -v $(pwd)/config/service_account.json:/app/service_account.json \
  gcr.io/$PROJECT_ID/disinformation-detection:latest
```

# For production deployment
```
./detection/deploy_prodv3.sh config/service_account.json --recreate-run
```
Can use the --recreate-ml flag for the ml server, but some commands in gpu_startup_script may need to be run manually, if not all (included those commetned out)

## testing the container
```
docker build -t gcr.io/$PROJECT_ID/disinformation-detection:latest -f run.dockerfile .
docker push gcr.io/$PROJECT_ID/disinformation-detection:latest

gcloud run deploy disinformation-detection     --image gcr.io/$PROJECT_ID/disinformation-detection:latest     --platform managed     --region asia-southeast1     --set-env-vars GOOGLE_CLOUD_PROJECT=$PROJECT_ID,GOOGLE_CLOUD_ZONE=asia-southeast1-c,GCE_INSTANCE_NAME=ml-instance     --allow-unauthenticated
```

### if the docker container in the ML Server isnt running properly
```
# sudo docker build -t ml-server:latest -f /tmp/cuda.dockerfile .
sudo docker run -d --gpus all -p 5000:5000 ml-server:latest
```

### commands to test each section
```
curl -X GET https://YOUR_PUBLIC_HTTPS_VALIDATION_CONTAINER/health

curl -X GET https://$(gcloud compute instances describe ml-instance --zone=asia-southeast1-c '--format=get(networkInterfaces[0].accessConfigs[0].natIP)'):5000/health

curl -X GET http://YOUR_PUBLIC_HTTPS_CUDA_CONTAINER:5000/health
curl -X POST http://YOUR_PUBLIC_HTTPS_CUDA_CONTAINER:5000/predict -H "Content-Type: application/json" -d '{"input":"Trump doesnt believe in Covid-19."}'

curl -X POST https://YOUR_PUBLIC_HTTPS_VALIDATION_CONTAINER/predict -H "Content-Type: application/json" -d '{"input":"Trump doesnt believe in Covid-19."}'
<!-- https://YOUR_PUBLIC_HTTPS_VALIDATION_CONTAINER -->
```

# For shutdown
```
./shutdown.sh config/service_account.json python-server-api-gcr

LMFE_MAX_CONSECUTIVE_WHITESPACES=2 python llm_handler.py
DEV=true python server.py
```

# to get the keys and cert
```
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout nginx-selfsigned.key -out nginx-selfsigned.crt -config openssl.cnf
sudo chown reisub-laptop nginx-selfsigned.*
```

# logging and debugging
```
gcloud projects get-iam-policy $PROJECT_ID
gcloud container images list --repository=gcr.io/$PROJECT_ID
gcloud run services describe disinformation-detection --region asia-southeast1

gcloud compute ssh --zone "asia-southeast1-c" "ml-instance" --project "$PROJECT_ID"
```

# make sure the firewall is setup properly
```
gcloud compute firewall-rules create allow-port-5000 \
    --network default \
    --direction INGRESS \
    --priority 1000 \
    --action ALLOW \
    --rules tcp:5000 \
    --source-ranges 0.0.0.0/0
```