CREATE DATABASE IF NOT EXISTS enernoc;

USE enernoc;

-- create SITES table for metadata sites
DROP TABLE IF EXISTS SITES;
CREATE TABLE SITES(
	id SMALLINT(6) NOT NULL,
	industry VARCHAR(100) NOT NULL,
	sub_industry VARCHAR(100) NOT NULL,
	sq_ft INT,
	lat VARCHAR(100),
	lng VARCHAR(100),
	time_zone VARCHAR(100) NOT NULL,
	tz_offset DATE NOT NULL
)ENGINE=InnoDB;
LOAD DATA LOCAL INFILE 'data/meta/all_sites.csv'
INTO TABLE SITES
FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' LINES TERMINATED BY '\n' 
IGNORE 1 LINES
(id,industry,sub_industry,sq_ft,lat,lng,time_zone,tz_offset);

ALTER TABLE SITES
ADD PRIMARY KEY(id); -- add primary key to make query on site faster

-- create CONSO table for consumption data 
DROP TABLE IF EXISTS CONSO;
CREATE TABLE CONSO(
	-- id SMALLINT(6) UNSIGNED NOT NULL AUTO_INCREMENT,
	-- PRIMARY KEY(id),
	timestamp INT(12),
	dttm_utc DATETIME NOT NULL,
	value DECIMAL(11,8),
	estimated TINYINT,
	anomaly VARCHAR(10),
	site SMALLINT(6) NOT NULL
)ENGINE=InnoDB;


-- DROP PROCEDURE IF EXISTS add_conso;
-- DELIMITER |
-- CREATE PROCEDURE add_conso()
-- BEGIN 
	-- DECLARE v_site_id SMALLINT(6); 
	-- DECLARE v_leave_loop TINYINT DEFAULT FALSE;
	-- DECLARE cursor_site CURSOR FOR
	-- SELECT site_id FROM SITES;
	-- DECLARE CONTINUE HANDLER FOR NOT FOUND
	-- BEGIN 
		-- SET v_leave_loop = TRUE;
	-- END;

	-- OPEN cursor_site;
	-- site_loop : LOOP
		-- FETCH cursor_site INTO v_site_id;
		-- IF v_leave_loop = TRUE THEN
			-- LEAVE site_loop;
		-- END IF;
		-- SELECT v_site_id AS "site";
		-- LOAD DATA LOCAL INFILE 'data/meta/6.csv'
		-- INTO TABLE CONSO
		-- FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' LINES TERMINATED BY '\n' 
		-- IGNORE 1 LINES
		-- (timestamp,dttm_utc,value,estimated,anomally);	
	-- END LOOP site_loop;
	-- CLOSE cursor_site;
-- END| 
-- DELIMITER ;
-- CALL add_conso();
