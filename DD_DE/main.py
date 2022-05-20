from random import random

from DD_DE import lti_system
from DD_DE import disturbance_estimator
from DD_DE import data_driven_predictor


def get_dist() -> float:
    return (random()-0.5)/10


def main():
    my_system = lti_system.LTISystem(0, 1, 1, get_dist)

    h_matrix = [[1,-1,0,-1],[0,1,1,0],[0,0,1,1],[1,-1,1,0]]
    my_predictor = data_driven_predictor.DDPredictor(h_matrix)

    print(my_predictor.h_matrix)
    print(my_predictor.h_matrix_inv)

    state_prediction = my_predictor.predict_state(my_system.x,1)
    print(state_prediction)

    # print(f"actual state: {my_system.x}")

    # print(f"Predicted state:  {my_predictor.predict_state(my_system.x,1)}")
    # my_system.next_step(1)
    # print(f"actual state: {my_system.x}")

    # my_system.next_step(2)
    # print(f"actual state: {my_system.x}")


if __name__ == "__main__":
    main()
