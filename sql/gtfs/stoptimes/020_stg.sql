INSERT IGNORE INTO stg_gtfs_stoptimes
(
SELECT
    CONCAT(`route_id`, '_', `agency_id`, '_', `route_type`) AS `gtfs_route_id`,
    `route_id`,
    `is_night`,
    `route_id`,
    `agency_id`,
    `route_url`,
    `route_desc`,
    `route_type`,
    `is_regional`,
    `route_color`,
    `route_long_name`,
    `route_short_name`,
    `route_text_color`,
    `is_substitute_transport`,
    `_sys_record_id`,
    `_sys_load_id`,
    `_sys_load_at`,
    `_sys_is_deleted`,
    `i`
FROM (
    SELECT
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.is_night')) AS `is_night`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.route_id')) AS `route_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.agency_id')) AS `agency_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.route_url')) AS `route_url`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.route_desc')) AS `route_desc`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.route_type')) AS `route_type`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.is_regional')) AS `is_regional`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.route_color')) AS `route_color`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.route_long_name')) AS `route_long_name`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.route_short_name')) AS `route_short_name`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.route_text_color')) AS `route_text_color`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.is_substitute_transport')) AS `is_substitute_transport`,
        `_sys_record_id`,
        `_sys_load_id`,
        `_sys_load_at`,
        `_sys_is_deleted`,
        `i`
    FROM src_gtfs_stoptimes
    ) a
);

TRUNCATE src_gtfs_routessrc_gtfs_stoptimes