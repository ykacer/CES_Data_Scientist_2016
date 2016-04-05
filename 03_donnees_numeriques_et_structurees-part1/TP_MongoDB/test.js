db = db.getSiblingDB("library");
db.dropDatabase();
document = ({ 
		Type : "Book",
       		Title : "Definitive Guide to MongoDB", 
	 	ISBN : "987-1-4302-3051-9",
		Publisher : "Apress",
		Author: ["Membrey, Peter", "Plugge, Eelco", "Hawkins, Tim" ]
	} )
db.media.insert(document);

db.media.insert({ 
			Type : "CD",
			Artist : "Nirvana",
			Title : "Nevermind",
			Tracklist : [
					{	 
						Track : "1 ",
						Title : "Smells like teen spirit ",
						Length : "5:02 "
					},
					{
						Track : "2 ",
						Title : "In Bloom ",
						Length : "4:15 "
					}
				]
	})
print("\n*** db.media.find():");
db.media.find().forEach( function(doc) {printjson(doc);} )

print("*** db.media.find( { Artist : 'Nirvana' } ):")
db.media.find({ 
		Artist : "Nirvana" 
		} 
	).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find( {Artist : 'Nirvana'}, {Title: 1} ):")
db.media.find(	{
			Artist : "Nirvana"
		}, 
		{
			Title: 1
		} 
	).forEach( function(doc) {printjson(doc)} )


print("\n*** db.media.find( {Artist : 'Nirvana'}, {Title: 0} ):")
db.media.find({
		Artist : "Nirvana"
	      },
	      {
		Title : 0
	      }).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find( { 'Tracklist.Title' : 'In Bloom' } )")
db.media.find({
		"Tracklist.Title" : "In Bloom"
	      }).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.findOne()");
printjson(db.media.findOne())

print("\n*** db.media.find().sort( { Title: 1 })");
db.media.find().sort({
			Title: 1 
		     }).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find().sort( { Title: -1 })");
db.media.find().sort({
			Title: -1
		     }).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find().limit( 10 )");
db.media.find().limit( 10 ).forEach( function(doc) {printjson(doc);} )


print("\n*** db.media.find().skip( 20 )");
db.media.find().skip( 20 ).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find().sort ( { Title : -1 } ).limit ( 10 ).skip ( 20 )");
db.media.find().sort ({
			Title : -1 
		      }).limit ( 10 ).skip ( 20 ).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.count()");
print(db.media.count())

print("\n*** db.media.find( { Publisher : 'Apress', Type: 'Book' } ).count()");
print(
db.media.find({
		Publisher : "Apress",
		Type : "Book"
		}).count()
)

print("\n*** db.media.find( { Publisher: 'Apress', Type:'Book' }).skip(2).count(true)");
print(
db.media.find({ 
			Publisher : "Apress",
			Type :"Book"
		}).skip(2).count(true)
)


document = ( {
		Type : "Book",
	 	Title : "Definitive Guide to MongoDB", 
	 	ISBN: "1-4302-3051-7",
		Publisher : "Apress",
	 	Author : ["Membrey, Peter","Plugge, Eelco","Hawkins, Tim"]
	} )
db.media.insert(document)

print("\nd*** b.media.distinct( 'Title')");
db.media.distinct( "Title").forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.distinct( 'ISBN')");
db.media.distinct ("ISBN").forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.group ( {key: {Title : true},initial : {Total : 0},reduce : function (items,prev) { prev.Total += 1 }} )")
db.media.group({
			key: 
			{
				Title : true
			},
			initial :
			{
				Total : 0
			},
			reduce : 
				function (item,prev) { prev.Total += 1 }
	} ).forEach( function(doc) {printjson(doc);} )

print("\n*** select sum(Tracklist.length) as Somme from media where type=' CD ' group by Title")
db.media.find({Tracklist : {$exists : true}}).forEach( function(doc) {
					if(doc.Tracklist instanceof Array){
						doc.Tracklist.forEach( function(el) {
							var str_Length = new String(el.Length);
							el.iso_length = ISODate("2012-01-01T00:0"+str_Length+"Z");
						})
					db.media.save(doc);
					}
		})

db.media.group({
			key:
			{
				"Tracklist.Title" : true
			},
			initial :
			{
				Somme : 0
			},
			reduce :
				function (item,prev) { prev.Somme += 1}, /*item.Tracklist.iso_length },*/
			cond :
			{
				Type : "CD"
			}
	} ).forEach( function(doc) {printjson(doc);} )


dvd = ( { Type : "DVD", Title : "Matrix, The", Released : 1999, Cast: ["KeanuReeves","Carry-Anne Moss","Laurence Fishburne","Hugo Weaving","Gloria Foster","JoePantoliano"] } )
db.media.insert(dvd)
dvd = ( { "Type" : "DVD", "Title" : "Toy Story 3", "Released" : 2010 } )
db.media.insert(dvd)
/* Insertion avec JavaScript */
function insertMedia( type, title, released){
db.media.insert({
"Type":type,
"Title":title,
"Released":released
});
}
insertMedia("DVD",   "Blade Runner",  1982 )

print("\n*** db.media.find ( { Released : {$gt : 2000} }, { 'Cast' : 0 } )")
db.media.find ( { 
		Released :
			{
				$gt : 2000
			}
		}, 
		{
			Cast : 0 
		}).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find( {Released : {$gte: 1990, $lt : 2010}}, { 'Cast' : 0 })")
db.media.find( {
		Released :
			{
				$gte: 1990, $lt : 2010
			}
		},
		{
			Cast : 0 
		}).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find( { Type : 'Book', Author: {$ne : 'Plugge, Eelco'}})")
db.media.find( {
			Type : "Book",
			Author: {$ne : "Plugge, Eelco"}
	})

print("\n*** db.media.find( {Released : {$in : ['1999','2008','2009'] } }, { 'Cast' :0 } )")
db.media.find({
		Released : 
			{
				$in : [1999,2008,2009]
			}
		},
		{
			"Cast" : 0
		}
	).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find( {Released : {$nin : ['1999','2008','2009'] },Type : 'DVD' }, { 'Cast' : 0 } )")
db.media.find({
		Released :
			{
				$nin : [1999,2008,2009]
			},
		Type : "DVD"
		},
		{
			"Cast" : 0
		}
	).forEach( function(doc) {printjson(doc);} )

/* $or */
print("\n*** db.media.find({ $or : [ { 'Title' : 'Toy Story 3' }, { 'ISBN' : '987-1-4302-3051-9' } ] } )")
db.media.find({ 
		$or : [ 
			{ 
				"Title" : "Toy Story 3" 
			},
			{ "ISBN" : "987-1-4302-3051-9"
			} 
		]
	}).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find({ 'Type' : 'DVD', $or : [ { 'Title' : 'Toy Story 3' },{ 'ISBN' : '987-1-4302-3051-9' } ] })")
db.media.find({ 
		"Type" : "DVD",
		$or : [ 
			{
				"Title" : "Toy Story 3"
			},
			{ 
				"ISBN" : "987-1-4302-3051-9"
			}
		]
	}).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find({'Title' : 'Matrix, The'}, {'Cast' : {$slice: 3}})")
db.media.find({
		"Title" : "Matrix, The"
		},
		{
			"Cast" : {$slice: 3}
		}).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find({'Title' : 'Matrix, The'}, {'Cast' : {$slice: -3}})")
db.media.find({
		"Title" : "Matrix, The"
		},
		{
			"Cast" : {$slice: -3}
		}).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find ( { Tracklist : {$size : 2} } )")
db.media.find ( { Tracklist : {$size : 2} } ).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find ( { Author : {$exists : true } } )")
db.media.find ( { Author : {$exists : true } } ).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.find ( { Author : {$exists : false } } )")
db.media.find ( { Author : {$exists : false } } ).forEach( function(doc) {printjson(doc);} )

/* creation index */
db.media.ensureIndex( { Title :1 } )
/* Index descendant */
db.media.ensureIndex( { Title :-1 } )
/* Index pour les objets incrustés (enbed object) */
db.media.ensureIndex( { "Tracklist.Title" : 1 } )
/* Forcer l’utilisation d’un index: hint() */
db.media.ensureIndex({ISBN: 1})

print("\n*** db.media.find( { ISBN: '987-1-4302-3051-9'} ) . hint ( { ISBN:-1 } )")
db.media.find( { ISBN: "987-1-4302-3051-9"} ) . hint ( { ISBN: 1 } ).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.getIndexes()")
db.media.getIndexes().forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.update( { 'Title' : 'Matrix, The'}, {'Type' : 'DVD', 'Title' : 'Matrix, The','Released' : '1999', 'Genre' : 'Action'}, true)")
db.media.update({
			"Title" : "Matrix, The"
		},
		{
			"Type" : "DVD",
			"Title" : "Matrix, The",
			"Released" : "1999",
			"Genre" : "Action"
		},
		 true)
db.media.find({Title : "Matrix, The"}).forEach( function(doc) {printjson(doc);} )

/* Ajout/suppression d’un attribut */
print("\n*** db.media.update ( { 'Title' : 'Matrix, The' }, {$set : { Genre : 'Sci-Fi' } } )")
db.media.update ({
			"Title" : "Matrix, The"
		},
		{
			$set : { Genre : "Sci-Fi" }
		})
db.media.find({Title : "Matrix, The"}).forEach( function(doc) {printjson(doc);} )

print("\n*** db.media.update ( {'Title': 'Matrix, The'}, {$unset : { 'Genre' : 1 } } )")
db.media.update ({
			"Title": "Matrix, The"
		},
		{
			$unset : { "Genre" : 1 }
		})
db.media.find({Title : "Matrix, The"}).forEach( function(doc) {printjson(doc);} )

db.media.remove( { "Title" : "Different Title" } )
/* Tous les documents */
db.media.remove({})
/* Toute la collection */
db.media.drop()
db.media.find().forEach( function(doc) {printjson(doc);} )
