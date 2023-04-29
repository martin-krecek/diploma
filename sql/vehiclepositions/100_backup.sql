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
    agency_name_real,
    agency_name_scheduled,
    sequence_id,
    start_timestamp,
    last_stop_arrival,
    last_stop_departure,
    last_stop_id,
    last_stop_sequence,
    last_stop_arrival_time,
    next_stop_id,
    next_stop_sequence,
    next_stop_arrival_time,
    next_stop_departure_time,
    is_canceled,
    origin_timestamp
FROM stg_vehiclepositions;