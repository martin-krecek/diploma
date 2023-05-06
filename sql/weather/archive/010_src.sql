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
            precipitation JSON PATH '$[1][0]',
            rain JSON PATH '$[2][0]',
            snowfall JSON PATH '$[3][0]',
            temperature JSON PATH '$[4][0]'
            )
    ) AS r
);
