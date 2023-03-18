INSERT IGNORE INTO stg_gtfs_stops
(
SELECT
    CONCAT(`stop_id`, '_', `stop_name`) AS `gtfs_stop_id`,
    `stop_id`,
    `zone_id`,
    `level_id`,
    `stop_name`,
    `location_type`,
    `platform_code`,
    `parent_station`,
    `wheelchair_boarding`,
    `_sys_record_id`,
    `_sys_load_id`,
    `_sys_load_at`,
    `_sys_is_deleted`,
    `i`
FROM (
    SELECT
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.stop_id')) AS `stop_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.zone_id')) AS `zone_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.level_id')) AS `level_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.stop_name')) AS `stop_name`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.location_type')) AS `location_type`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.platform_code')) AS `platform_code`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.parent_station')) AS `parent_station`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.wheelchair_boarding')) AS `wheelchair_boarding`,
        `_sys_record_id`,
        `_sys_load_id`,
        `_sys_load_at`,
        `_sys_is_deleted`,
        `i`
    FROM src_gtfs_stops
    ) a
);
