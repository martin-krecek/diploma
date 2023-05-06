CREATE TABLE IF NOT EXISTS pre_weather_forecast (
    _sys_record_id INT NOT NULL AUTO_INCREMENT,
    jdoc JSON,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted BOOLEAN,
    CONSTRAINT id PRIMARY KEY (_sys_record_id, _sys_load_at)
);

CREATE TABLE IF NOT EXISTS src_weather_forecast (
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted BOOLEAN,
    i INT,
    time JSON,
    precipitation VARCHAR(255),
    rain JSON,
    snowfall JSON,
    temperature JSON,
    CONSTRAINT id PRIMARY KEY (_sys_record_id,i,_sys_load_at)
);

CREATE TABLE IF NOT EXISTS stg_weather_forecast (
    weather_forecast_id VARCHAR(255) NOT NULL,
    time_ts TIMESTAMP,
    time VARCHAR(255),
    precipitation VARCHAR(255),
    temperature_2m VARCHAR(255),
    snowfall VARCHAR(255),
    rain VARCHAR(255),
    _sys_record_id INT,
    _sys_load_id INT,
    _sys_load_at TIMESTAMP,
    _sys_is_deleted VARCHAR(255),
    i INT,
    PRIMARY KEY (weather_forecast_id)
  );
