INSERT IGNORE INTO src_parking_measurements
(
SELECT
    _sys_record_id,
    _sys_load_id,
    _sys_load_at,
    _sys_is_deleted,
    r.*
FROM
    pre_parking_measurements,
    JSON_TABLE(
        jdoc,
        '$[*]'
        COLUMNS (
            i FOR ORDINALITY,
            jdoc JSON PATH '$[0]'
            )
    ) AS r
);

TRUNCATE pre_parking_measurements