-- out_parking_measurements
CREATE OR REPLACE VIEW `out_parking_measurements` AS 
select 
  `stg_parking_measurements`.`parking_measurement_id` AS `parking_measurement_id`, 
  `stg_parking_measurements`.`source` AS `source`, 
  `stg_parking_measurements`.`source_id` AS `source_id`, 
  `stg_parking_measurements`.`parking_id` AS `parking_id`, 
  `stg_parking_measurements`.`date_modified` AS `date_modified`, 
  `stg_parking_measurements`.`total_spot_number` AS `total_spot_number`, 
  `stg_parking_measurements`.`occupied_spot_number` AS `occupied_spot_number`, 
  `stg_parking_measurements`.`available_spot_number` AS `available_spot_number`,
  'stg' AS `_sys_source_table` 
from 
  `stg_parking_measurements` 
where 
  (
    `stg_parking_measurements`.`date_modified` > '2023-03-18'
  ) 
union 
select 
  `stg_parking_measurements_history`.`parking_measurement_id` AS `parking_measurement_id`, 
  `stg_parking_measurements_history`.`source` AS `source`, 
  `stg_parking_measurements_history`.`source_id` AS `source_id`, 
  `stg_parking_measurements_history`.`parking_id` AS `parking_id`, 
  `stg_parking_measurements_history`.`date_modified` AS `date_modified`, 
  `stg_parking_measurements_history`.`total_spot_number` AS `total_spot_number`, 
  `stg_parking_measurements_history`.`occupied_spot_number` AS `occupied_spot_number`, 
  `stg_parking_measurements_history`.`available_spot_number` AS `available_spot_number`,
  'stg_history' AS `_sys_source_table` 
from 
  `stg_parking_measurements_history` 
where 
  (
    `stg_parking_measurements_history`.`date_modified` < '2023-03-18'
  );

-- out_weather
CREATE OR REPLACE VIEW `out_weather` AS 
select 
  `stg_weather_forecast`.`weather_forecast_id` AS `weather_id`, 
  `stg_weather_forecast`.`time_ts` AS `time_ts`,
  SUBDATE(`stg_weather_forecast`.`time_ts`,8) AS `time_ts_shifted`,
  `stg_weather_forecast`.`time` AS `time`, 
  `stg_weather_forecast`.`precipitation` AS `precipitation`, 
  `stg_weather_forecast`.`temperature` AS `temperature`, 
  `stg_weather_forecast`.`snowfall` AS `snowfall`, 
  `stg_weather_forecast`.`rain` AS `rain`,
  `stg_weather_forecast`.`_sys_load_at` AS `_sys_load_at`,
  'stg_forecast' AS `_sys_source_table` 
from 
  `stg_weather_forecast` 
where 
  (
    `stg_weather_forecast`.`time_ts` > SUBDATE(CURDATE(),7)
  ) 
  and _sys_load_at = (SELECT MAX(_sys_load_at) FROM stg_weather_forecast)
union 
select 
  `stg_weather_archive`.`weather_archive_id` AS `weather_id`, 
  `stg_weather_archive`.`time_ts` AS `time_ts`,
  `stg_weather_archive`.`time_ts` AS `time_ts_shifted`,
  `stg_weather_archive`.`time` AS `time`, 
  `stg_weather_archive`.`precipitation` AS `precipitation`, 
  `stg_weather_archive`.`temperature` AS `temperature`, 
  `stg_weather_archive`.`snowfall` AS `snowfall`, 
  `stg_weather_archive`.`rain` AS `rain`,
  `stg_weather_archive`.`_sys_load_at` AS `_sys_load_at`,
  'stg_archive' AS `_sys_source_table` 
from 
  `stg_weather_archive` 
where 
  (
    `stg_weather_archive`.`time_ts` <= SUBDATE(CURDATE(),7)
  )
  
-- out_vehiclepositions
CREATE OR REPLACE VIEW `out_vehiclepositions` AS 
select 
  `stg_vehiclepositions`.`vehicleposition_id` AS `vehicleposition_id`,
  `stg_vehiclepositions`.`geometry_coordinates` AS `geometry_coordinates`,
  `stg_vehiclepositions`.`gtfs_trip_id` AS `gtfs_trip_id`,
  `stg_vehiclepositions`.`gtfs_route_id` AS `gtfs_route_id`,
  `stg_vehiclepositions`.`gtfs_route_type` AS `gtfs_route_type`,
  `stg_vehiclepositions`.`start_timestamp` AS `start_timestamp`,
  `stg_vehiclepositions`.`actual` AS `actual`,
  `stg_vehiclepositions`.`last_stop_sequence` AS `last_stop_sequence`,
  `stg_vehiclepositions`.`last_stop_arrival_time` AS `last_stop_arrival_time`,
  'stg' AS `_sys_source_table`
from 
  `stg_vehiclepositions`;