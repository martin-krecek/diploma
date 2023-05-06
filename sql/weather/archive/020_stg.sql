INSERT IGNORE INTO stg_weather_archive
(
SELECT
    CONCAT(`time`, '_', `rain`, '_', `temperature`) AS `weather_archive_id`,
    `time_ts`,
    `time`,
    `rain`,
    `snowfall`,
    `temperature`,
    `_sys_record_id`,
    `_sys_load_id`,
    `_sys_load_at`,
    `_sys_is_deleted`,
    `i`
FROM (
    SELECT
        `_sys_record_id`,
        `_sys_load_id`,
        `_sys_load_at`,
        `_sys_is_deleted`,
        `i`,
        `time` AS `time_ts`,
        REPLACE(`time`, '"', '') AS `time`,
        `rain`,
        `snowfall`,
        `temperature`
    FROM src_weather_archive
    ) AS r
);