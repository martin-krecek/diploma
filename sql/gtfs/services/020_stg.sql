INSERT IGNORE INTO stg_gtfs_services
(
SELECT
    CONCAT(`service_id`, '_', `start_date`, '_', `end_date`) AS `gtfs_service_id`,
    `service_id`,
    `monday`,
    `tuesday`,
    `wednesday`,
    `thursday`,
    `friday`,
    `saturday`,
    `sunday`,
    `start_date`,
    `end_date`,
    `created_at`,
    `created_by`,
    `updated_at`,
    `updated_by`,
    `create_batch_id`,
    `update_batch_id`,
    `_sys_record_id`,
    `_sys_load_id`,
    `_sys_load_at`,
    `_sys_is_deleted`,
    `i`
FROM (
    SELECT
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.service_id')) AS `service_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.monday')) AS `monday`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.tuesday')) AS `tuesday`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.wednesday')) AS `wednesday`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.thursday')) AS `thursday`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.friday')) AS `friday`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.saturday')) AS `saturday`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.sunday')) AS `sunday`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.start_date')) AS `start_date`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.end_date')) AS `end_date`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.created_at')) AS `created_at`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.created_by')) AS `created_by`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.updated_at')) AS `updated_at`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.updated_by')) AS `updated_by`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.create_batch_id')) AS `create_batch_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.update_batch_id')) AS `update_batch_id`,
        `_sys_record_id`,
        `_sys_load_id`,
        `_sys_load_at`,
        `_sys_is_deleted`,
        `i`
    FROM src_gtfs_services
    ) a
);
