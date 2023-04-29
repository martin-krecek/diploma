TRUNCATE stg_parking_spaces_backup;

INSERT INTO stg_parking_spaces_backup
SELECT
    *
FROM stg_parking_spaces;