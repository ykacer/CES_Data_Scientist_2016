populatePhones = function(area,start,stop) 
{for(var i=start; i < stop; i++){
	var country = 1 + ((Math.random() * 8) << 0);
	var num = (country * 1e10) + (area * 1e7) + i;
	db.testcollection.insert({
		_id: num,
		components: {
			country: country,
			area: area,
			prefix: (i * 1e-4) << 0,
			number: i,
			testtext:"Because of the nature of MongoDB, many of the more traditional functions that a DB Administrator would perform are not required. Creating new databases collections and new fields on the server are no longer necessary, as MongoDB will create these elements on-the-fly as you access them. Therefore, for the vast majority of cases managing databases and schemas is not required."
		},
		display: "+" + country + " " + area + "-" + i
	});
       }
}
