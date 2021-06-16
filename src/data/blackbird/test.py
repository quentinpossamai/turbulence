import matplotlib.pyplot as plt
import pickle
from src.data.blackbird.blackbird_loading import FLIGHT_LIST, load_flight

processing_time = pickle.load(open("/Users/quentin/phd/turbulence/processing_time.pkl", "rb"))
sorted_time = sorted(processing_time, key=lambda x: processing_time[x])

plt.plot([processing_time[e] for e in sorted_time])
plt.axvline(159)
plt.grid()

flight = sorted_time[-1]
flight_name, flight_yaw, flight_speed = flight.split("/")[:-1]
data = load_flight(flight_name, flight_yaw, flight_speed)

time = data[:, 0]
state = data[:, 1:13]
motor = data[:, 13:17]
imu = data[:, 17:23]
next_time = data[:, 23:24]
dstate = data[:, 24:36]
next_state = data[:, 36:]

plt.plot(time, state[:, 0:3])
plt.plot(time, state[:, 3:6])
plt.plot(time, state[:, 6:9])
plt.plot(time, state[:, 9:12])

fig, axs = plt.subplots(2, 1, figsize=(19.2, 10.8))
axs[0].plot(time, state[:, 9:12])
axs[1].plot(time, motor)
for ax in axs:
    ax.grid()
    ax.legend()
