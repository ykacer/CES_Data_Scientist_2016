# High-résolution satellite images for human density predicition

We show here after how to build a human density map prediction for a certain country using one of our built predictive model. In this tutorial, we use Japan as example.

## Bring metadata
We need a list of Japanese cities with their surfaces (and populations, not mandatory, only useful to compute error on prediction). [Wikipedia](https://en.wikipedia.org/wiki/List_of_cities_in_Japan) brings us such a table taken from Japan Institute from 2007.
Just copy paste the table in a csv file and rename header for surface (from 'area' to 'SURFACE'), header for population (from 'population' to 'PMUN13') and header for city name (from 'city' to 'LIBMIN'). This is just a pure formatting convention. The file should look like this :

<pre>
LIBMIN				 	Japanese 	Prefecture 	PMUN13	 	SURFACE

Nagoya					名古屋市	 Aichi 		 2,283,289 	 326.45 	

Toyohashi 				豊橋市		Aichi 		377,045 	261.35 

Okazaki 				岡崎市		Aichi 		371,380 	387.24
</pre>


Then, store the file as `data/Japan/population_surface_Japan.csv`.

## Bring coordinates cities
We now need to localize precisely each of the cities using `geopy`, a Python module to query geolocalisations. 
Use `code/python/geo/locator.py` file as follows :

`python code/python/geo/locator.py data/Japan/population_surface_japan.csv data/Japan/population_surface_coordonnees_japan.csv`

This operation can take several hours as the `GoogleV3` geolocator used, doesn't allow too much queries and some pauses are imposed by Google.
You should get a new file `data/Japan/population_surface_coordonnees_japan.csv` containing now all metadata needed : name, surface, population, latitude and longitude for each city. 

## Bring Landsat-8 satellite images data
We now need to query images from USGS website. Be sure that you get a (free) logon to export your query in a csv file.
First, make a polygon containing Japan and put a data time range big enough to contain a summer period :
<p align="center">
  <img src="data/Japon/japon-selection.png" width="450"/>
</p>

Be sure that you enable a large amount of data results into Results Options window (from 100 to 500 could be enough) :
<p align="center">
  <img src="data/Japon/japon-selection2.png" width="450"/>
</p>

Then, specify the nature of datasets needed (sensors OLI/TIRS)
<p align="center">
  <img src="data/Japon/japon-datasets.png" width="450"/>
</p>

Add some conditions to your queries in terms of cloud covering (<20%) and day/night selection (day):
<p align="center">
  <img src="data/Japon/japon-criteria.png" width="450"/>
</p>

Finally get your results and export it as a CSV file
<p align="center">
  <img src="data/Japon/japon-results.png" width="450"/>
</p>

You should get a zip file whose name has the form of `LANDSAT_8_XXXXXX.zip`, unzip the containing into `data/Japon` folder.
You have now all information to get Landsat-8 datasets but we need to remove redundant ones (same path,row). The following bash file do this automatically, keeping the less cloudyness datasets for a certain path,row :

`./code/utils/landsat/landsat-8-clean data/Japon/LANDSAT_8_XXXXXX.csv`

You should obtain a lighter file `data/Japon/LANDSAT_8_XXXXXX_clean.csv`

We can now verify that this last file contains datasets that covers all the territory. The following bash file download thumbnails of datasets and project them in map:

`./code/utils/landsat/landsat-8-draw data/Japon/LANDSAT_8_XXXXXX_clean.csv`

 The resulting image is `data/Japon/covering-selection.png` and you should verify that the territory is well covered (if no, enlarge the time range, or increase the cloudyness from 20% to 30%)
<p align="center">
  <img src="data/Japon/covering-selection.png" width="450"/>
</p>

## Compute NDVI (vegetal indice) histogram for each city

Now, simply run :

 `python code/python/core/ndvi_features.py data/Japon/population_surface_coordonnees.csv data/Japon/LANDSAT_8_XXXXXX.csv 13`

the number '13' correspond to the header name 'PMUN13' we talked about when formatting metadata.
The script will download each datasets and form corresponding NDVI image.
You should obtain `data/Japan/ndvi_features.py` containing 1024-size NDVI vector for each city, the explanatory variables.

## Density prediction

We are now able to finally get our density prediction using Neural Network built model for example:

`python code/python/ndvi_test_classification.py data/Japon/ndvi_features.csv model_classification/Neural_Network_Classification-oversampling/Neural_Network_Classification-oversampling.pkl`

A new folder `data/Japon/test/Neural_Network_Classification-oversampling` will then be automatically created, that contains `density_ground_truth.png` (available only if true population is provided in medatada csv file) and `density_classification.png` for the predictions made by the model. Note that here, the ground truth correspond to 2007 Japan Government Institute while the prediction corresponds to 2015 landsat-8 images.
