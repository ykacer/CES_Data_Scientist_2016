set profiling = 1;

SELECT "*** calculate the sum LD for the 100 sites (timestamp interval: 5 minutes)";
SELECT site,SUM(value) AS "LD sum" FROM CONSO GROUP BY site;
SELECT "*** time execution profile"
SHOW PROFILE;

SELECT "*** calculate the average LD by sector of activity (timestamp interval: 5 minutes)";
SELECT industry,AVG(value) AS "LD average" FROM CONSO
INNER JOIN SITES ON CONSO.site = SITES.id
GROUP BY industry;
SELECT "*** time execution profile"
SHOW PROFILE;

SELECT "*** calculate the total LD for the 100 sites (timestamp interval : a week)";
SELECT site,WEEK(dttm_utc) AS week,SUM(value) AS "LD average" FROM CONSO GROUP BY site,week;
SELECT "*** time execution profile"
SHOW PROFILE;

SELECT "*** calculate the average LD by sector of activity (timestamp interval : a week)";
SELECT industry,WEEK(dttm_utc) AS week,AVG(value) AS "LD average" FROM CONSO
INNER JOIN SITES ON CONSO.site = SITES.id
GROUP BY industry,week;
SELECT "*** time execution profile"
SHOW PROFILE;

