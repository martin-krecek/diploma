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
            time JSON PATH '$[0][0]',
            rain JSON PATH '$[0][1]',
            snowfall JSON PATH '$[0][2]',
            temperature JSON PATH '$[0][3]'
            )
    ) AS r
);
