INSERT IGNORE INTO stg_meteosensors
(
SELECT
    CONCAT(`id`, '_', `last_updated`) AS `meteosensor_capture_id`,
    `id`,
    `name`,
    `district`,
    `humidity`,
    `updated_at`,
    `wind_speed`,
    `last_updated`,
    `wind_direction`,
    `air_temperature`,
    `road_temperature`,
    `geo_location`,
    `_sys_record_id`,
    `_sys_load_id`,
    `_sys_load_at`,
    `_sys_is_deleted`,
    `i`
FROM (
    SELECT
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.id')) AS `id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.name')) AS `name`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.district')) AS `district`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.humidity')) AS `humidity`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.updated_at')) AS `updated_at`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.wind_speed')) AS `wind_speed`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.last_updated')) AS `last_updated`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.wind_direction')) AS `wind_direction`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.air_temperature')) AS `air_temperature`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.road_temperature')) AS `road_temperature`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.geometry.coordinates')) AS `geo_location`,
        `_sys_record_id`,
        `_sys_load_id`,
        `_sys_load_at`,
        `_sys_is_deleted`,
        `i`
    FROM src_meteosensors
    ) a
);

TRUNCATE src_meteosensors;