# install requirements
pip install -r requirements.txt

# mount bucket to data/gcs
gcloud auth application-default login
fusermount -u /home/anyafries/animal-facilities/data/gcs
gcsfuse --implicit-dirs cs325b-animal-facilities data/gcs

# check the mounting was successful
ls data/gcs/data/top_20