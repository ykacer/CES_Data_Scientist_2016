db = db.getSiblingDB("enernoc")

print("*** select number of light industrial in New York:");
var debut1 = new Date();
print(db.sites.find(
			{INDUSTRY : "Light Industrial"},
			{TIME_ZONE : "America/New_York"}
		).count()
);
var fin1 = new Date();
print("----> execution time:",(fin1-debut1)/1000,"s");

print("*** select the site that has the biggest annual average consumption:");
var debut2 = new Date();
biggest = db.sites.aggregate([
		{$project : 
			{ 
				site : "$SITE_ID",
				avg_conso : {$avg:"$CONSO.value"}
			}
		},
		{$sort :
			{
				avg_conso :-1
			}
		},
		{$limit : 1}
	]).next()
print("site",biggest.site,"(annual average consu:",biggest.avg_conso,")");
var fin2 = new Date();
print("----> execution time:",(fin2-debut2)/1000,"s");


print("*** select average consumption during winter (between January and Mars) for site 766:");
var debut3 = new Date();
d = ISODate("2012-04-01T00:00:00Z")
winter = db.getCollection("766").aggregate([
		{$match : 
			{
				iso_date : {$lt:d}
			}
		},
		{$group :
			{
				_id : null,
				avg_consu : {$avg:"$value"}
			}
		}
	]).next()
print(winter.avg_consu)
var fin3 = new Date();
print("----> execution time:",(fin3-debut3)/1000,"s");


print("*** select the peak consumption among all northern sites (LAT>37):");
var debut4 = new Date();
peak1 = db.sites.aggregate([
		{ $match : 
			{
				LAT : { $gt : 37 }
			}
	       	},
		{$project :
			{
				SITE_ID : 1,
     				"CONSO.value" : 1, 
      				"CONSO.iso_date" : 1
			}
		},
		{$unwind : "$CONSO"},
		{$sort :
			{
				"CONSO.value":-1
			}
		},
		{$limit : 1}
	]).next()
print(peak1.CONSO.value,"in site",peak1.SITE_ID,"on",peak1.CONSO.iso_date.toDateString());
var fin4 = new Date();
print("----> execution time:",(fin4-debut4)/1000,"s");


print("*** select the consumption among all southern sites (LAT<37):");
var debut5 = new Date();
peak2 = db.sites.aggregate([
		{$match: 
			{
				LAT : { $lt : 37 }
			} 
		},
		{$project :
			{
				SITE_ID : 1, 
      				"CONSO.value" : 1, 
      				"CONSO.iso_date" : 1
			}
		},
		{$unwind : "$CONSO"},
		{$sort : 
			{
				"CONSO.value" : -1
			}
		},
		{$limit : 1}
	]).next()
print(peak2.CONSO.value,"in site",peak2.SITE_ID,"on",peak2.CONSO.iso_date.toDateString());
var fin5 = new Date();
print("----> execution time:",(fin5-debut5)/1000,"s");


print("*** Calculate the sum LD for the 100 sites (timestamp interval: 5 minutes");
var debut6 = new Date();
db.getCollection("sites").aggregate([
		{$project : 
			{ 
				site : "$SITE_ID",
				sum_consu : {$sum : "$CONSO.value"}
			}
		},
		{$sort:
			{
				sum_consu : -1
			}
		}
	]).forEach( function(doc) 
		{
			print("site:",doc.site,"\tLD sum:",doc.sum_consu)
		}
	);
var fin6 = new Date();
print("----> execution time:",(fin6-debut6)/1000,"s");


print("*** Calculate the average LD by sector of activity (timestamp interval : 5 minutes)");
var debut7 = new Date();
db.getCollection("sites").aggregate([
		{$unwind : "$CONSO"},
		{$group : 
			{
				_id:"$INDUSTRY",
				avg_consu:{$avg:"$CONSO.value"}
			}
		},
		{$project : 
			{
				"_id" : 1,
				"avg_consu" : 1
			}
		}
	]).forEach( function(doc) 
		{
			print("industry:",doc._id,"\tavg LD:",doc.avg_consu)
		}
	);
var fin7 = new Date();
print("----> execution time:",(fin7-debut7)/1000,"s");


print("*** Calculate the total LD for the 100 sites (timestamp interval: a week)");
var debut8 = new Date();
db.sites.aggregate([
		{$unwind : "$CONSO"},
		{$group :
		       	{
				 _id:
				{
					site_id : "$SITE_ID",
					week : {$week : "$CONSO.iso_date"}
				},
				sum_consu :
				{
					$sum : "$CONSO.value"
				}
			 }
		},
		{$sort :
		       	{
				 _id:1
			}
		}
	]).forEach( function(doc) 
		{
			print("site:",doc._id.site_id,"\tweek:",doc._id.week,"\tsum LD:",doc.sum_consu)
		}
	);
var fin8 = new Date();
print("----> execution time:",(fin8-debut8)/1000,"s");



print("*** Calculate the average LD by sector of activity (timestamp interval: a week)");
var debut9 = new Date();
db.sites.aggregate([
		{$unwind : "$CONSO"},
		{$group : 
			{
				_id :
				{
					industry:"$INDUSTRY",week: {$week:"$CONSO.iso_date"}
				},
				avg_consu : 
				{
					$avg:"$CONSO.value"
				}
			}
		},
		{$sort:
			{
				_id:1
			}
		}
	]).forEach( function(doc) 
		{
			print("industry:",doc._id.industry,"\tweek:",doc._id.week,"\tsum LD:",doc.avg_consu)
		}
	);
var fin9 = new Date();
print("----> execution time:",(fin9-debut9)/1000,"s");
