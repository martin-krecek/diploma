TRUNCATE stg_parking_measurements_backup;

INSERT INTO stg_parking_measurements_backup
SELECT
    *
FROM stg_parking_measurements;