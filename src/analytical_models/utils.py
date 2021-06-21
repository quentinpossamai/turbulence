import os
import pickle


def load_flights(mode):
    assert mode in ["validation", "test", "train"]

    ## Dataset location
    datadir = "/storage/Datasets/BlackBird/"  # Dataset location on cluster
    if not os.path.isdir(datadir):
        datadir = "/Volumes/Samsung_T5/BlackBird/"  # Dataset location (Steeven)
    if not os.path.isdir(datadir):
        datadir = "???"  # TODO : mettre le path vers les données sur le Mac de Quentin
    assert os.path.isdir(datadir)

    ## Load test/val flights
    with open(datadir + mode + ".txt", "r") as file:
        flight_list = file.read().split("\n")[:-1]
    flights = {}
    for flight in flight_list:
        flight_data = pickle.load(open(f"{datadir}intermediate/{flight}data.pkl", 'rb'))
        flights[flight] = flight_data
    return flights
