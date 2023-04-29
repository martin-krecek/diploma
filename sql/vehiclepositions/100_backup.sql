TRUNCATE stg_vehiclepositions_backup;

INSERT INTO stg_vehiclepositions_backup
SELECT
    vehicleposition_id,
    geometry_coordinates,
    gtfs_trip_id,
    gtfs_route_id,
    gtfs_route_type,
    gtfs_trip_headsign,
    gtfs_trip_short_name,
    gtfs_route_short_name,
    sequence_id,
    start_timestamp,
    last_stop_arrival,
    last_stop_departure,
    last_stop_id,
    last_stop_sequence,
    last_stop_arrival_time,
    origin_timestamp,
    _sys_record_id,
    _sys_load_at,
    i
FROM stg_vehiclepositions;