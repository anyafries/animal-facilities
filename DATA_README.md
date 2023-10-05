========================================================
README for cs325b-animal-facilities
========================================================

Overview:
---------
This GCP bucket contains datasets from various satellite sources including NAIP and PlanetScope
The datasets included in this bucket are:

1. NAIP RGBN Data
   - https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ
2. PlanetScope scenes
3. California animal facilities data
   - https://data.ca.gov/dataset/surface-water-water-quality-regulated-facility-information/resource/c16335af-f2dc-41e6-a429-f19edba5b957

Structure:
----------
The bucket is structured as follows:

- top_20/				(Data for top 20 poultry and dairy farms in headcount)
    - facility_data/
       - animal_facilities_bbox.csv		(Bounding boxes used to extract sat imagery)
       - animal_report.csv				(Full css from California data portal)
       - dairy_20.csv					(Filtered df of top 20 dairy farms)
       - poultry_20.csv					(Filtered df of top 20 poultry farms)
   - naip/								(All NAIP imagery for a given bounding box)
      - *.tif							(NAIP RGBN image named <animal_report.csv idx>_<month>_<day>_<year>.tif)
   - psscenes/							(All PlanetScope imagery corresponding to NAIP imagery)
      - <animal_report.csv idx>_<year>	(All PlanetScope scenes from April 8-15 of given year)
         - *_AnalyticMS_SR_clip_reproject.tif	(RGBN PlanetScope scenes)
         - *_udm2_clip_reproject.tif.		(Additional metadata https://developers.planet.com/docs/data/udm-2/)


- all_farms/				(Data for all poultry and dairy farms)
   - naip/					(All NAIP imagery for a given bounding box)
      - *.tif				(NAIP RGBN image named <animal_report.csv idx>_<month>_<day>_<year>.tif)

** Not all farms will have NAIP imagery as some farms didn't have coverage

Access:
-------
To access the data in this bucket, use the following gsutil command:

    gsutil -m cp -r gs://cs325b-animal_facilities/[FOLDER_NAME] [DESTINATION]

For example, to copy the satellite data to your local machine, use:

    gsutil -m cp -r gs://cs325b-animal_facilities/data/top_20/naip [DESTINATION]

Note: Replace `[BUCKET_NAME]` with the name of this GCP bucket and `[DESTINATION]` with your desired local destination.