import datetime
from sklearn.preprocessing import MinMaxScaler

# db parameters
user = 'martin'
pw = 'admin'
db = 'mysql'    
host = '34.77.219.108'

table = 'parking_measurements'
parking_id = 'tsk-534017'

current_date = datetime.date.today()
one_day_ago = current_date - datetime.timedelta(days=1)
eight_days_ago = current_date - datetime.timedelta(days=8)
seven_days_forward = current_date + datetime.timedelta(days=7)

query_predict = f"SELECT date_modified, occupied_spot_number FROM out_{table} WHERE parking_id = '{parking_id}' AND date_modified >= '{eight_days_ago}' AND date_modified < '{one_day_ago}';"
query_predict_weather = f"SELECT time_ts_shifted as time_ts, temperature, precipitation FROM out_weather WHERE time_ts >= '{current_date}' AND time_ts < '{seven_days_forward}';"

print(query_predict)
print(query_predict_weather)