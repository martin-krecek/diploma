CREATE TABLE IF NOT EXISTS pre_parking_measurements_history (
    _sys_record_id INT NOT NULL AUTO_INCREMENT,
    jdoc JSON,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted BOOLEAN,
    CONSTRAINT id PRIMARY KEY (_sys_record_id, _sys_load_at)
);

CREATE TABLE IF NOT EXISTS src_parking_measurements_history (
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted BOOLEAN,
    i INT,
    jdoc JSON,
    CONSTRAINT id PRIMARY KEY (_sys_record_id,i,_sys_load_at)
);

CREATE TABLE IF NOT EXISTS stg_parking_measurements_history (
    parking_measurement_id VARCHAR(255) NOT NULL,
    source VARCHAR(255),
    source_id VARCHAR(255),
    parking_id VARCHAR(255),
    date_modified TIMESTAMP,
    total_spot_number VARCHAR(255),
    closed_spot_number VARCHAR(255),
    occupied_spot_number VARCHAR(255),
    available_spot_number VARCHAR(255),
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted VARCHAR(255),
    i INT,
    PRIMARY KEY (parking_measurement_id)
  );
