var w = 600;
var h = 600;
var dataset = []
var svg = d3.select("body").append("svg").attr("width",w).attr("height",h);

d3.tsv("data/france.tsv").row(function (d,i) {
	return {
		codePostal: +d["Postal Code"], // mettre un + pour ne pas recupérer l'attribut en string
		inseeCode: +d.inseecode,
		place: d.place,
		longitude: +d.x,
		latitude: +d.y,
		population: +d.population,
		densite: +d.density
	};
}).get(function(error,rows) {
	console.log("Loaded " + rows.length + " rows");
	if (rows.length > 0) {
		console.log("First row : ", rows[0])
		console.log("Last row : ", rows[rows.length-1])
		console.log("42nd row : ", rows[42])
	}
	x = d3.scale.linear().domain(d3.extent(rows, function(row) { return row.longitude; })).range([0, w]);
	y = d3.scale.linear().domain(d3.extent(rows, function(row) { return row.latitude; })).range([h, 0]);
	dataset = rows;
	draw();
});



function draw() {
	svg.selectAll("rect")
		.data(dataset)
		.enter()
		.append("rect")
		.attr("width",1)
		.attr("height",1)
		.attr("x",function(d) { return x(d.longitude) })
		.attr("y",function(d) { return y(d.latitude) })
		.on("mouseover",handleMouseOver)
		.on("mouseout",handleMouseOut);
}

function handleMouseOver(d,i) {	
	d3.select(this).attr({});
	svg.append("text").attr({
		id: "t"+"-"+i,
		x : function() { return x(d.longitude); },
		y : function() { return y(d.latitude); }
		})
	.text(function() {
		return d.place ;
	})
}

function handleMouseOut(d, i) {
	d3.select(this).attr({});
	d3.select("#t"+"-"+i).remove();
}
