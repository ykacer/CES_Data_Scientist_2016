db = db.getSiblingDB('enernoc')
var n = site_id.toString();
db.getCollection(n).remove({anomaly :{$ne:""}});
db.getCollection(n).find().forEach(function(cs){ cs.iso_date = new Date(cs.timestamp*1000); db.getCollection(n).save(cs);})
db.sites.update({'SITE_ID':site_id},{'$set' : {"CONSO" : db.getCollection(n).find().toArray()}});
