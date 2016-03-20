db = db.getSiblingDB('geodb')

print("\n*** ajouter une colonne iso_date dont la valeur est la conversion du timestamp en date");
db.earthquakes.find().forEach( function(doc) {
	doc.properties.iso_date = new Date(doc.properties.time);
	db.earthquakes.save(doc);
});
print(tojson(db.earthquakes.findOne()));

print("\n*** parser la chaine de catactères 'properties.types' en tableau de mots 'properties.types_as_array'")
db.earthquakes.find().forEach( function(doc) {
	doc.properties.types_as_array = doc.properties.types.split(",");
	db.earthquakes.save(doc);
})
print(tojson(db.earthquakes.findOne()));

print("\n*** eliminer tous les elements vides des tableaux 'properties.types_as_array'")
db.earthquakes.update({},
		{
			$pullAll : {
				"properties.types_as_array" : [""]
			}
		},
		{
			multi : true
		}
	)
print(tojson(db.earthquakes.findOne()));

print("\n*** donner le nombre de documents dont la liste 'types_as_array' contient 'geoserve' et 'tectonic-summary'");
var n = db.earthquakes.find({
		$and : [
			{
			"properties.types_as_array" : {
							$in : ['geoserve']
						}
			},
			{
			"properties.types_as_array" : {
							$in : ['tectonic-summary']
						}
    			
			}
		]
	}).count();
print(n);

print("\n*** créer un champ 'depth' egale à la troisième coordonnée de 'properties.coordinates'");
db.earthquakes.find().forEach( function(doc) {
			doc.depth = doc.geometry.coordinates[2];		
			db.earthquakes.save(doc);
	});
db.earthquakes.update({},
		{
			$pop : {
				"geometry.coordinates" : 1
			}
		},
		{
			multi : true	
		}
		)
print(tojson(db.earthquakes.findOne()));

print("\n*** ajouter un index de type 2D sur les attributs 'geometry.coordinates'");
db.earthquakes.createIndex( { geometry : "2dsphere" } );
db.earthquakes.getIndexes().forEach( function(doc) {print(tojson(doc))} );

print("\n*** trouver les tremblements de terre proche de -3,984,48,724 (1000km)");
db.earthquakes.find({
	geometry:
		{ $near :
			{
				$geometry : { 
					type: "Point", 
					coordinates : [-3.984,48.724]
				},
				$minDistance : 0,
				$maxDistance : 1000000
			}
		}
	}).forEach( function(doc) {print(tojson(doc))} ); 
