var w = 600;
var h = 600;
var dataset = []
var svg = d3.select("body").append("svg").attr("width",w).attr("height",h);
var p = d3.select("body").append("p").style("position","absolute");
var c = d3.hsl(0,0,0);

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
	
	dens = d3.scale.linear().domain(d3.extent(rows, function(row) { return row.densite; })).range([0, 1]);
	
	popu = d3.scale.linear().domain(d3.extent(rows, function(row) { return row.population; })).range([0, 1]);

	dataset = rows;
	
	draw();
});

function draw() {
	//svg.selectAll("rect")
	//	.data(dataset)
	//	.enter()
	//	.append("rect")
	//	.attr("width",function(d) {return 50*popu(d.population)})
	//	.attr("height",function(d) {return 50*popu(d.population)})
	//	.attr("x",function(d) { return x(d.longitude) })
	//	.attr("y",function(d) { return y(d.latitude) })
	//	.attr("fill",function(d) { return  d3.hsl(60*dens(d.densite),100,0.5).toString() })
	//	.on("mouseover",handleMouseOver)
	//	.on("mouseout",handleMouseOut);
	svg.selectAll("circle")
		.data(dataset)
		.enter()
		.append("circle")
		.attr("cx", function(d) { return x(d.longitude)})
		.attr("cy", function(d) { return y(d.latitude) })
		.attr("r" , function(d) { return 50*popu(d.population)})
		.style("stroke",function(d) { return  d3.hsl(60+180*dens(d.densite),100,0.5).toString() })
		.style("fill", "none")
		.on("mouseover",handleMouseOver)
		.on("mouseout",handleMouseOut);
}

function handleMouseOver(d) {	
	p.style("top", d3.select(this).attr("y")).style("left",d3.select(this).attr("x"));
	p.text("commune : "+d.place).append("p").text("population : "+d.population).append("p").text("densité : "+d.densite)
}

function handleMouseOut(d, i) {
	d3.select(this).attr({});
	d3.select("#t"+"-"+i).remove();
}
