INSERT IGNORE INTO stg_vehiclepositions
(
SELECT
    CONCAT(`gtfs_trip_id`, '_', `gtfs_route_id`, '_', `gtfs_route_type`, '_', `gtfs_route_short_name`, '_', `gtfs_trip_short_name`, '_', `last_stop_arrival_time`) AS vehicleposition_id,
    `type`,
    `geometry_type`,
    `geometry_coordinates`,        
    `cis_line_id`,
    `cis_trip_number`,
    `gtfs_trip_id`,
    `gtfs_route_id`,
    `gtfs_route_type`,
    `gtfs_trip_headsign`,
    `gtfs_trip_short_name`,
    `gtfs_route_short_name`,
    `agency_name_real`,
    `agency_name_scheduled`,
    `sequence_id`,
    `vehicle_type_id`,
    `vehicle_type_description_cs`,
    `vehicle_type_description_en`,
    `air_conditioned`,
    `start_timestamp`,
    `origin_route_name`,
    `wheelchair_accessible`,
    `vehicle_registration_number`,
    `actual`,
    `last_stop_arrival`,
    `last_stop_departure`,
    `speed`,
    `bearing`,
    `tracking`,
    `last_stop_id`,
    `last_stop_sequence`,
    `last_stop_arrival_time`,
    `last_stop_departure_time`,
    `next_stop_id`,
    `next_stop_sequence`,
    `next_stop_arrival_time`,
    `next_stop_departure_time`,
    `is_canceled`,
    `state_position`,
    `origin_timestamp`,
    `shape_dist_traveled`,
    `_sys_record_id`,
    `_sys_load_id`,
    `_sys_load_at`,
    `_sys_is_deleted`,
    i
FROM (
    SELECT
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.type')) AS `type`,

        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.geometry.type')) AS `geometry_type`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.geometry.coordinates')) AS `geometry_coordinates`,
        
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.cis.line_id')) AS `cis_line_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.cis.trip_number')) AS `cis_trip_number`,

        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.gtfs.trip_id')) AS `gtfs_trip_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.gtfs.route_id')) AS `gtfs_route_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.gtfs.route_type')) AS `gtfs_route_type`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.gtfs.trip_headsign')) AS `gtfs_trip_headsign`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.gtfs.trip_short_name')) AS `gtfs_trip_short_name`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.gtfs.route_short_name')) AS `gtfs_route_short_name`,

        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.agency_name.real')) AS `agency_name_real`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.agency_name.scheduled')) AS `agency_name_scheduled`,

        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.sequence_id')) AS `sequence_id`,

        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.vehicle_type.id')) AS `vehicle_type_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.vehicle_type.description_cs')) AS `vehicle_type_description_cs`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.vehicle_type.description_en')) AS `vehicle_type_description_en`,

        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.air_conditioned')) AS `air_conditioned`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.start_timestamp')) AS `start_timestamp`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.origin_route_name')) AS `origin_route_name`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.wheelchair_accessible')) AS `wheelchair_accessible`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.trip.vehicle_registration_number')) AS `vehicle_registration_number`,

        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.delay.actual')) AS `actual`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.delay.last_stop_arrival')) AS `last_stop_arrival`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.delay.last_stop_departure')) AS `last_stop_departure`,

        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.speed')) AS `speed`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.bearing')) AS `bearing`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.tracking')) AS `tracking`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.last_stop.id')) AS `last_stop_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.last_stop.sequence')) AS `last_stop_sequence`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.last_stop.arrival_time')) AS `last_stop_arrival_time`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.last_stop.departure_time')) AS `last_stop_departure_time`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.next_stop.id')) AS `next_stop_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.next_stop.sequence')) AS `next_stop_sequence`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.next_stop.arrival_time')) AS `next_stop_arrival_time`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.next_stop.departure_time')) AS `next_stop_departure_time`,

        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.is_canceled')) AS `is_canceled`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.state_position')) AS `state_position`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.origin_timestamp')) AS `origin_timestamp`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_position.shape_dist_traveled')) AS `shape_dist_traveled`,        

        `_sys_record_id`,
        `_sys_load_id`,
        `_sys_load_at`,
        `_sys_is_deleted`,
        i
    FROM src_vehiclepositions
    ) a
);

TRUNCATE TABLE src_vehiclepositions;
