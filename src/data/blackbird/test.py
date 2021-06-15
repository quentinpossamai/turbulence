import matplotlib.pyplot as plt
import pickle
from src.data.blackbird.blackbird_loading import FLIGHT_LIST, load_flight

processing_time = pickle.load(open("/Users/quentin/phd/turbulence/processing_time.pkl", "rb"))
sorted_time = sorted(processing_time, key=lambda x: processing_time[x])

plt.plot([processing_time[e] for e in sorted_time])
plt.axvline(159)
plt.grid()

for e in sorted_time[160:]:
    print(e)
    flight_name, flight_yaw, flight_speed = e.split("/")[:-1]
    data = load_flight(flight_name, flight_yaw, flight_speed)
    break
