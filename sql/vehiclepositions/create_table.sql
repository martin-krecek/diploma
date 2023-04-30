CREATE TABLE IF NOT EXISTS pre_vehiclepositions (
    _sys_record_id INT NOT NULL AUTO_INCREMENT,
    jdoc JSON,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted BOOLEAN,
    CONSTRAINT id PRIMARY KEY (_sys_record_id, _sys_load_at)
);

CREATE TABLE IF NOT EXISTS src_vehiclepositions (
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted BOOLEAN,
    i INT,
    jdoc JSON,
    CONSTRAINT id PRIMARY KEY (_sys_record_id,i,_sys_load_at)
);

CREATE TABLE IF NOT EXISTS stg_vehiclepositions (
    vehicleposition_id VARCHAR(255) NOT NULL,
    type VARCHAR(255),
    geometry_type VARCHAR(255),
    geometry_coordinates VARCHAR(255),
    cis_line_id VARCHAR(255),
    cis_trip_number VARCHAR(255),
    gtfs_trip_id VARCHAR(255),
    gtfs_route_id VARCHAR(255),
    gtfs_route_type VARCHAR(255),
    gtfs_trip_headsign VARCHAR(255),
    gtfs_trip_short_name VARCHAR(255),
    gtfs_route_short_name VARCHAR(255),
    agency_name_real VARCHAR(255),
    agency_name_scheduled VARCHAR(255),
    sequence_id VARCHAR(255),
    vehicle_type_id VARCHAR(255),
    vehicle_type_description_cs VARCHAR(255),
    vehicle_type_description_en VARCHAR(255),
    air_conditioned VARCHAR(255),
    start_timestamp TIMESTAMP,
    origin_route_name VARCHAR(255),
    wheelchair_accessible VARCHAR(255),
    vehicle_registration_number VARCHAR(255),
    actual VARCHAR(255),
    last_stop_arrival VARCHAR(255),
    last_stop_departure VARCHAR(255),
    speed VARCHAR(255),
    bearing VARCHAR(255),
    tracking VARCHAR(255),
    last_stop_id VARCHAR(255),
    last_stop_sequence VARCHAR(255),
    last_stop_arrival_time TIMESTAMP,
    last_stop_departure_time TIMESTAMP,
    next_stop_id VARCHAR(255),
    next_stop_sequence VARCHAR(255),
    next_stop_arrival_time TIMESTAMP,
    next_stop_departure_time TIMESTAMP,
    is_canceled VARCHAR(255),
    state_position VARCHAR(255),
    origin_timestamp TIMESTAMP,
    shape_dist_traveled VARCHAR(255),
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted VARCHAR(255),
    i INT,
    CONSTRAINT id PRIMARY KEY (_sys_record_id,i,_sys_load_at,gtfs_trip_id)
  );

CREATE TABLE IF NOT EXISTS stg_vehiclepositions_backup (
    vehicleposition_id VARCHAR(255) NOT NULL,
    geometry_coordinates VARCHAR(255),
    gtfs_trip_id VARCHAR(255),
    gtfs_route_id VARCHAR(255),
    gtfs_route_type VARCHAR(255),
    gtfs_trip_headsign VARCHAR(255),
    gtfs_trip_short_name VARCHAR(255),
    gtfs_route_short_name VARCHAR(255),
    sequence_id VARCHAR(255),
    start_timestamp TIMESTAMP,
    actual VARCHAR(255),
    last_stop_arrival VARCHAR(255),
    last_stop_departure VARCHAR(255),
    last_stop_id VARCHAR(255),
    last_stop_sequence VARCHAR(255),
    last_stop_arrival_time TIMESTAMP,
    _sys_record_id INT,
    _sys_load_at TIMESTAMP,
    i INT,
    CONSTRAINT id PRIMARY KEY (_sys_record_id,i,_sys_load_at,gtfs_trip_id)
  );