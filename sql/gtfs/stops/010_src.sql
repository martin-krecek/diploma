INSERT IGNORE INTO src_gtfs_stops
(
SELECT
    _sys_record_id,
    _sys_load_id,
    _sys_load_at,
    _sys_is_deleted,
    r.*
FROM
    pre_gtfs_stops,
    JSON_TABLE(
        jdoc,
        '$[*]'
        COLUMNS (
            i FOR ORDINALITY,
            jdoc JSON PATH '$[0]'
            )
    ) AS r
);

TRUNCATE pre_gtfs_stops