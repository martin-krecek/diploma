CREATE TABLE IF NOT EXISTS pre_gtfs_stoptimes (
_sys_record_id INT NOT NULL AUTO_INCREMENT,
jdoc JSON,
_sys_load_id INT,
_sys_load_at TIMESTAMP,
_sys_is_deleted BOOLEAN,
CONSTRAINT id PRIMARY KEY (_sys_record_id, _sys_load_at)
);

CREATE TABLE IF NOT EXISTS src_gtfs_stoptimes (
_sys_record_id INT,
_sys_load_id INT,
_sys_load_at TIMESTAMP,
_sys_is_deleted BOOLEAN,
i INT,
jdoc JSON,
CONSTRAINT id PRIMARY KEY (_sys_record_id,i,_sys_load_at)
);
