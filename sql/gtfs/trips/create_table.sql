CREATE TABLE IF NOT EXISTS pre_gtfs_trips (
_sys_record_id INT NOT NULL AUTO_INCREMENT,
jdoc JSON,
_sys_load_id INT,
_sys_load_at TIMESTAMP,
_sys_is_deleted BOOLEAN,
CONSTRAINT id PRIMARY KEY (_sys_record_id, _sys_load_at)
);

CREATE TABLE IF NOT EXISTS src_gtfs_trips (
_sys_record_id INT,
_sys_load_id INT,
_sys_load_at TIMESTAMP,
_sys_is_deleted BOOLEAN,
i INT,
jdoc JSON,
CONSTRAINT id PRIMARY KEY (_sys_record_id,i,_sys_load_at)
);

CREATE TABLE IF NOT EXISTS stg_gtfs_trips (
    gtfs_trips_id VARCHAR(255) NOT NULL,
    trip_id VARCHAR(255),
    block_id VARCHAR(255),
    route_id VARCHAR(255),
    shape_id VARCHAR(255),
    service_id VARCHAR(255),
    exceptional VARCHAR(255),
    direction_id VARCHAR(255),
    bikes_allowed VARCHAR(255),
    trip_headsign VARCHAR(255),
    trip_short_name VARCHAR(255),
    trip_operation_type VARCHAR(255),
    wheelchair_accessible VARCHAR(255),
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted VARCHAR(255),
    i INT,
    PRIMARY KEY (gtfs_trips_id)
  );
