INSERT IGNORE INTO stg_weather_forecast
(
SELECT
    CONCAT(`time`, '_', `rain`, '_', `temperature`) AS `weather_forecast_id`,
    `time_ts`,
    `time`,
    `precipitation`,
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
        `precipitation`,
        `rain`,
        `snowfall`,
        `temperature`
    FROM src_weather_forecast
    ) AS r
);

TRUNCATE TABLE src_weather_forecast;