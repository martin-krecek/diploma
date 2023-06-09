INSERT IGNORE INTO stg_parking_spaces
(
SELECT
    CONCAT(`space_id`, '_', `name`, '_', `available_spots_last_updated`) AS `parking_space_id`,
    `space_id`,
    `name`,
    `source`,
    `postal_code`,
    `address_region`,
    `street_address`,
    `address_country`,
    `address_locality`,
    `address_formatted`,
    `category`,
    `type`,
    `coordinates`,
    `valid_to`,
    `source_id`,
    `tariff_id`,
    `zone_type`,
    `valid_from`,
    `area_served`,
    `parking_type`,
    `data_provider`,
    `date_modified`,
    `total_spot_number`,
    `ios_app_payment_url`,
    `web_app_payment_url`,
    `available_spots_number`,
    `android_app_payment_url`,
    `available_spots_last_updated`,
    `_sys_record_id`,
    `_sys_load_id`,
    `_sys_load_at`,
    `_sys_is_deleted`,
    `i`
FROM (
    SELECT
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.id')) AS `space_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.name')) AS `name`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.source')) AS `source`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.address.postal_code')) AS `postal_code`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.address.address_region')) AS `address_region`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.address.street_address')) AS `street_address`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.address.address_country')) AS `address_country`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.address.address_locality')) AS `address_locality`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.address.address_formatted')) AS `address_formatted`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.category')) AS `category`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.centroid.type')) AS `type`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.centroid.coordinates')) AS `coordinates`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.valid_to')) AS `valid_to`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.source_id')) AS `source_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.tariff_id')) AS `tariff_id`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.zone_type')) AS `zone_type`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.valid_from')) AS `valid_from`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.area_served')) AS `area_served`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.parking_type')) AS `parking_type`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.data_provider')) AS `data_provider`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.date_modified')) AS `date_modified`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.total_spot_number')) AS `total_spot_number`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.ios_app_payment_url')) AS `ios_app_payment_url`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.web_app_payment_url')) AS `web_app_payment_url`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.available_spots_number')) AS `available_spots_number`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.android_app_payment_url')) AS `android_app_payment_url`,
        TRIM(BOTH '"' FROM JSON_EXTRACT(`jdoc`, '$.properties.available_spots_last_updated')) AS `available_spots_last_updated`,
        `_sys_record_id`,
        `_sys_load_id`,
        `_sys_load_at`,
        `_sys_is_deleted`,
        `i`
    FROM src_parking_spaces
    ) a
);

TRUNCATE src_parking_spaces