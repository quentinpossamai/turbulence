from src.drone_models.analytical_models.analytical_model import DynamicalModel
from src.metrics import load_blackbird_flights
import torch


def identify_aero():
    model = DynamicalModel()
    flights = load_blackbird_flights("train")

    ## Load flights in a big vector column
    states, future_states, command = [], [], []
    for f in flights.keys():
        s = torch.FloatTensor(flights[f]["state"])
        motor = torch.FloatTensor(flights[f]["motor"])
        states.append(s[:-1])
        future_states.append(s[1:])
        command.append(motor[:-1])
    states = torch.cat(states, dim=0)
    future_states = torch.cat(future_states, dim=0)
    command = torch.cat(command, dim=0)[:, [1, 0, 3, 2]]
    B = states.shape[0]

    prediction = model(states, command)
    y = (future_states - prediction)[:, 7:]
    A = torch.zeros(B, 6, 12)
    v, w = states[:, 7:10], states[:, 10:]
    norm_v, norm_w = torch.sqrt((v ** 2).sum(-1)), torch.sqrt((w ** 2).sum(-1))
    for i in range(3):
        A[:, i, i] = norm_v * v[:, i]
        A[:, i, i + 3] = v[:, i]
        A[:, i + 3, i + 6] = norm_w * w[:, i]
        A[:, i + 3, i + 9] = w[:, i]

    A = A.view(-1, 12)
    y = y.reshape(-1)

    theta = torch.pinverse(A) @ y
    print(theta)


if __name__ == '__main__':
    identify_aero()
