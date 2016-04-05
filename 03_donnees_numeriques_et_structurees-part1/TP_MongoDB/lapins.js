db = db.getSiblingDB("elevage")
db.dropDatabase()
db.lapins.insert({
	nom : "Leny",
	genre : "F",
	ville : "Lyon",
	regime : [
		'carotte',
		'courgette'
		],
	poids : 4,
	taille : 20
})
db.lapins.insert({
	nom : "Bunny",
	genre : "H",
	ville : "Paris",
	poids : 3
})
db.lapins.insert({
	nom : "Olto",
	genre : "H",
	ville : "Paris",
	regime : [
		'raisin',
		'carotte',
		'salade'
		],
	poids : 5,
	taille : 25
})

print("\n*** Trouver les lapins males:")
db.lapins.find({genre:"H"}).forEach( function(doc) {print(tojson(doc))} );

print("\n*** Trouver les lapins qui pesent plus de 4kg et qui aiment les carottes");
db.lapins.find({poids:{$gt:4},regime:{$in:['carotte']}}).forEach( function(doc) {print(tojson(doc))} );

print("\n*** Trouver les lapins qui aiment les courgettes ou ls raisins ou qui n'ont pas de champ ville")
db.lapins.find({
		$or : [
			{
				regime : {
						$in : ['courgette','raisin']
					}
			},
			{
				ville : {
						$exists : false
					}
			}
		]
	}).forEach( function(doc) {print(tojson(doc))} );

print("\n*** Trouver les lapins qui n\'aiment pas les salades")
db.lapins.find({
		regime : {
				$nin : ['salade'] 
			}
		}).forEach(function(doc) {print(tojson(doc))} );

print("\n*** rajouter un champ pays à Bunny")
db.lapins.update({
			nom : "Bunny"
		},
		{
			$set : { pays : "France" }
		})
db.lapins.find({nom : "Bunny"}).forEach( function(doc) {print(tojson(doc))} )

print("\n*** Supprimer le champ taille s'il existe de tous les documents")
db.lapins.update({
			taille : { $exists : true }
		},
		{
			$unset : { taille : "" }
		},
		{
			multi : true
		})
db.lapins.find().forEach( function(doc) {print(tojson(doc))} )
db.dropDatabase()



