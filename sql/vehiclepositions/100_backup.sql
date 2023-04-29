TRUNCATE stg_vehiclepositions_backup;

INSERT INTO stg_vehiclepositions_backup
SELECT
    *
FROM stg_vehiclepositions;