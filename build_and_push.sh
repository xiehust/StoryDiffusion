#!/bin/bash
set -v
set -e

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the region name. 

region=$(aws configure get region)
profile="default"


# Get the account number associated with the current IAM credentials
account=$(aws sts --profile $profile get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi

inference_image=story-diffusion-inference-api
inference_fullname=${account}.dkr.ecr.${region}.amazonaws.com/${inference_image}:latest

# If the repository doesn't exist in ECR, create it.
aws --profile ${profile} ecr describe-repositories --repository-names "${inference_image}" --region ${region} || aws --profile ${profile} ecr create-repository --repository-name "${inference_image}" --region ${region}

if [ $? -ne 0 ]
then
    aws --profile ${profile} ecr create-repository --repository-name "${inference_image}" --region ${region}
fi

# Get the login command from ECR and execute it directly
aws --profile ${profile} ecr get-login-password --region $region | docker login --username AWS --password-stdin $account.dkr.ecr.$region.amazonaws.com

aws --profile ${profile} ecr set-repository-policy \
    --repository-name "${inference_image}" \
    --policy-text "file://ecr-policy.json" \
    --region ${region}

# Build the docker image locally with the image name and then push it to ECR
# with the full name.

docker build -t ${inference_image} -f Dockerfile . 

docker tag ${inference_image} ${inference_fullname}

docker push ${inference_fullname}