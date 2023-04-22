CREATE TABLE IF NOT EXISTS pre_meteosensors (
    _sys_record_id INT NOT NULL AUTO_INCREMENT,
    jdoc JSON,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted BOOLEAN,
    CONSTRAINT id PRIMARY KEY (_sys_record_id, _sys_load_at)
);

CREATE TABLE IF NOT EXISTS src_meteosensors (
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted BOOLEAN,
    i INT,
    jdoc JSON,
    CONSTRAINT id PRIMARY KEY (_sys_record_id,i,_sys_load_at)
);

CREATE TABLE IF NOT EXISTS stg_meteosensors (
    meteosensor_capture_id VARCHAR(255),
    id VARCHAR(255),
    name VARCHAR(255),
    district VARCHAR(255),
    humidity VARCHAR(255),
    updated_at TIMESTAMP,
    wind_speed VARCHAR(255),
    last_updated VARCHAR(255),
    wind_direction VARCHAR(255),
    air_temperature VARCHAR(255),
    road_temperature VARCHAR(255),
    geo_location VARCHAR(255),
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted VARCHAR(255),
    i INT,
    PRIMARY KEY (meteosensor_capture_id)
  );
