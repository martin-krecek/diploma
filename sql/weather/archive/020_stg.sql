INSERT IGNORE INTO stg_weather_archive
(
SELECT
    CONCAT(`time`, '_', `rain`, '_', `temperature`) AS `weather_archive_id`,
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
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.hourly.time')) AS `time`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.hourly.rain')) AS `rain`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.hourly.snowfall')) AS `snowfall`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.hourly.temperature_2m')) AS `temperature`,
        `_sys_record_id`,
        `_sys_load_id`,
        `_sys_load_at`,
        `_sys_is_deleted`,
        `i`
    FROM src_weather_archive
    ) a
);

--TRUNCATE src_weather_archive