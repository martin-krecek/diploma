CREATE TABLE IF NOT EXISTS pre_parking_spaces (
    _sys_record_id INT NOT NULL AUTO_INCREMENT,
    jdoc JSON,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted BOOLEAN,
    CONSTRAINT id PRIMARY KEY (_sys_record_id, _sys_load_at)
);

CREATE TABLE IF NOT EXISTS src_parking_spaces (
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted BOOLEAN,
    i INT,
    jdoc JSON,
    CONSTRAINT id PRIMARY KEY (_sys_record_id,i,_sys_load_at)
);

CREATE TABLE IF NOT EXISTS stg_parking_spaces (
    parking_space_id VARCHAR(255),
    space_id VARCHAR(255),
    name VARCHAR(255),
    source VARCHAR(255),
    postal_code VARCHAR(255),
    address_region VARCHAR(255),
    street_address VARCHAR(255),
    address_country VARCHAR(255),
    address_locality VARCHAR(255),
    address_formatted VARCHAR(255),
    category VARCHAR(255),
    type VARCHAR(255),
    coordinates VARCHAR(255),
    valid_to VARCHAR(255),
    source_id VARCHAR(255),
    tariff_id VARCHAR(255),
    zone_type VARCHAR(255),
    valid_from VARCHAR(255),
    area_served VARCHAR(255),
    parking_type VARCHAR(255),
    data_provider VARCHAR(255),
    date_modified VARCHAR(255),
    total_spot_number VARCHAR(255),
    ios_app_payment_url VARCHAR(255),
    web_app_payment_url VARCHAR(255),
    available_spots_number VARCHAR(255),
    android_app_payment_url VARCHAR(255),
    available_spots_last_updated TIMESTAMP,
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted VARCHAR(255),
    i INT,
    PRIMARY KEY (parking_space_id)
  );
