CREATE TABLE IF NOT EXISTS pre_gtfs_routes (
    _sys_record_id INT NOT NULL AUTO_INCREMENT,
    jdoc JSON,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted BOOLEAN,
    CONSTRAINT id PRIMARY KEY (_sys_record_id, _sys_load_at)
);

CREATE TABLE IF NOT EXISTS src_gtfs_routes (
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted BOOLEAN,
    i INT,
    jdoc JSON,
    CONSTRAINT id PRIMARY KEY (_sys_record_id,i,_sys_load_at)
);

CREATE TABLE IF NOT EXISTS stg_gtfs_routes (
    gtfs_route_id VARCHAR(255) NOT NULL,
    route_id VARCHAR(255),
    is_night VARCHAR(255),
    agency_id VARCHAR(255),
    route_url VARCHAR(255),
    route_desc VARCHAR(255),
    route_type VARCHAR(255),
    is_regional VARCHAR(255),
    route_color VARCHAR(255),
    route_long_name VARCHAR(255),
    route_short_name VARCHAR(255),
    route_text_color VARCHAR(255),
    is_substitute_transport VARCHAR(255),
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted VARCHAR(255),
    i INT,
    PRIMARY KEY (gtfs_route_id)
  );
