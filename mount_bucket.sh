#!/bin/bash

# Set variables
BUCKET_NAME="af1-vm"
SERVICE_ACCOUNT_JSON="login.json"
MOUNT_POINT="/data/gcs"

# Update the system
sudo apt-get update && sudo apt-get upgrade -y

# Install gcsfuse
echo "deb http://packages.cloud.google.com/apt gcsfuse-`lsb_release -c -s` main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y gcsfuse

# Setup environment for gcsfuse to use the service account
export GOOGLE_APPLICATION_CREDENTIALS=${SERVICE_ACCOUNT_JSON}

# Create a mount point
sudo mkdir -p ${MOUNT_POINT}
sudo chmod a+w ${MOUNT_POINT}

# Mount the bucket in read-only mode
gcsfuse --implicit-dirs ${BUCKET_NAME} ${MOUNT_POINT} --key-file ${SERVICE_ACCOUNT_JSON} -o ro

# Make sure the mount persists across reboots
echo "${BUCKET_NAME} ${MOUNT_POINT} gcsfuse _netdev,allow_other,implicit_dirs,ro 0 0" | sudo tee -a /etc/fstab
