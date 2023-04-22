INSERT IGNORE INTO stg_parking_measurements_history
(
SELECT
    CONCAT(`source_id`, '_', `parking_id`, '_', `date_modified`) AS `parking_measurement_id`,
    `source`,
    `source_id`,
    `parking_id`,
    `date_modified`,
    `total_spot_number`,
    `closed_spot_number`,
    `occupied_spot_number`,
    `available_spot_number`,
    `_sys_record_id`,
    `_sys_load_id`,
    `_sys_load_at`,
    `_sys_is_deleted`,
    `i`
FROM (
    SELECT
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.source')) AS `source`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.source_id')) AS `source_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.parking_id')) AS `parking_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.date_modified')) AS `date_modified`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.total_spot_number')) AS `total_spot_number`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.closed_spot_number')) AS `closed_spot_number`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.occupied_spot_number')) AS `occupied_spot_number`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.available_spot_number')) AS `available_spot_number`,
        `_sys_record_id`,
        `_sys_load_id`,
        `_sys_load_at`,
        `_sys_is_deleted`,
        `i`
    FROM src_parking_measurements_history
    ) a
);
