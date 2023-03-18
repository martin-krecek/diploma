INSERT IGNORE INTO src_gtfs_stoptimes
(
SELECT
    _sys_record_id,
    _sys_load_id,
    _sys_load_at,
    _sys_is_deleted,
    r.*
FROM
    pre_gtfs_stoptimes,
    JSON_TABLE(
        jdoc,
        '$[*]'
        COLUMNS (
            i FOR ORDINALITY,
            jdoc JSON PATH '$[0]'
            )
    ) AS r
);
