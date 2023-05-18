import datetime
# Get the current date
current_date = datetime.date.today()
# Calculate the offset to Monday (0 represents Monday, 1 represents Tuesday, and so on)
offset_to_monday = (current_date.weekday() - 0) % 7
# Calculate the date of Monday
monday_date = current_date - datetime.timedelta(days=offset_to_monday)

table = 'parking_measurements'
parking_id = 'tsk-534012'

query = f"SELECT date_modified, occupied_spot_number FROM out_{table} WHERE parking_id = '{parking_id}' AND date_modified >= '2021-05-03' AND date_modified < '{monday_date}';"
query_weather = f"SELECT time_ts, temperature, precipitation FROM out_weather WHERE time_ts >= '2021-05-03' AND time_ts < '{monday_date}';"

print(monday_date)

print(query)
print(query_weather)