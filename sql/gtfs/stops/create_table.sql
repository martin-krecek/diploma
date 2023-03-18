CREATE TABLE IF NOT EXISTS pre_gtfs_stops (
_sys_record_id INT NOT NULL AUTO_INCREMENT,
jdoc JSON,
_sys_load_id INT,
_sys_load_at TIMESTAMP,
_sys_is_deleted BOOLEAN,
CONSTRAINT id PRIMARY KEY (_sys_record_id, _sys_load_at)
);

CREATE TABLE IF NOT EXISTS src_gtfs_stops (
_sys_record_id INT,
_sys_load_id INT,
_sys_load_at TIMESTAMP,
_sys_is_deleted BOOLEAN,
i INT,
jdoc JSON,
CONSTRAINT id PRIMARY KEY (_sys_record_id,i,_sys_load_at)
);

CREATE TABLE IF NOT EXISTS stg_gtfs_stops (
    gtfs_stop_id VARCHAR(255) NOT NULL,
    stop_id VARCHAR(255),
    zone_id VARCHAR(255),
    level_id VARCHAR(255),
    stop_name VARCHAR(255),
    location_type VARCHAR(255),
    platform_code VARCHAR(255),
    parent_station VARCHAR(255),
    wheelchair_boarding VARCHAR(255),
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted VARCHAR(255),
    i INT,
    PRIMARY KEY (gtfs_stop_id)
  );
