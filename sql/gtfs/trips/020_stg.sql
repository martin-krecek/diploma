INSERT IGNORE INTO stg_gtfs_trips
(
SELECT
    CONCAT(`trip_id`, '_', `route_id`, '_', `shape_id`, '_', `service_id`) AS `gtfs_trip_id`,
    `trip_id`,
    `block_id`,
    `route_id`,
    `shape_id`,
    `service_id`,
    `exceptional`,
    `direction_id`,
    `bikes_allowed`,
    `trip_headsign`,
    `trip_short_name`,
    `trip_operation_type`,
    `wheelchair_accessible`,
    `_sys_record_id`,
    `_sys_load_id`,
    `_sys_load_at`,
    `_sys_is_deleted`,
    `i`
FROM (
    SELECT
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.trip_id')) AS `trip_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.block_id')) AS `block_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.route_id')) AS `route_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.shape_id')) AS `shape_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.service_id')) AS `service_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.exceptional')) AS `exceptional`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.direction_id')) AS `direction_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.bikes_allowed')) AS `bikes_allowed`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.trip_headsign')) AS `trip_headsign`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.trip_short_name')) AS `trip_short_name`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.trip_operation_type')) AS `trip_operation_type`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.wheelchair_accessible')) AS `wheelchair_accessible`,
        `_sys_record_id`,
        `_sys_load_id`,
        `_sys_load_at`,
        `_sys_is_deleted`,
        `i`
    FROM src_gtfs_trips
    ) a
);

TRUNCATE src_gtfs_trips