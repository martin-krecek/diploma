INSERT IGNORE INTO src_vehiclepositions
(
SELECT
    _sys_record_id,
    _sys_load_id,
    _sys_load_at,
    _sys_is_deleted,
    r.*
FROM
    pre_vehiclepositions,
    JSON_TABLE(
        jdoc,
        '$[*]'
        COLUMNS (
            i FOR ORDINALITY,
            jdoc JSON PATH '$[0]'
            )
    ) AS r
);
