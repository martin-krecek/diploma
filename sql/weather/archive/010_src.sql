INSERT IGNORE INTO src_weather_archive
(
SELECT
    _sys_record_id,
    _sys_load_id,
    _sys_load_at,
    _sys_is_deleted,
    r.*
FROM
    pre_weather_archive,
    JSON_TABLE(
        jdoc,
        '$[*]'
        COLUMNS (
            i FOR ORDINALITY,
            time JSON PATH '$.hourly.time',
            rain JSON PATH '$.hourly.rain',
            snowfall JSON PATH '$.hourly.snowfall',
            temperature JSON PATH '$.hourly.temperature'
            )
    ) AS r
);

--TRUNCATE pre_weather_archive