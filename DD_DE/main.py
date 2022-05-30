from random import random

from DD_DE import lti_system
from DD_DE import data_driven_predictor
from disturbance_estimator import gaussian_process



def get_dist() -> float:
    return (random()-0.5)/10


def main():
    my_system = lti_system.LTISystem(x=0, A=1, B=1, get_dist=get_dist)

    h_matrix = [[1, -1, 0, -1], [0, 1, 1, 0], [0, 0, 1, 1], [1, -1, 1, 0]]
    my_predictor = data_driven_predictor.DDPredictor(h_matrix)

    disturbance_estimator = gaussian_process.GaussianProcess()

    print(f"actual state: {my_system.x}")

    for u in [1,0,-3,4,-5,6]:
        print(f"\n\nk = {my_system.k}:")

        predicted_state = my_predictor.predict_state(my_system.x,u)

        print(f"Predicted state:  {my_predictor.predict_state(my_system.x,u)}")
        my_system.next_step(u)
        print(f"actual state: {my_system.x}")

        delta_x = my_system.x - predicted_state

        disturbance_estimator.add_delta_x(my_system.k,delta_x)

    disturbance_estimator.plot_distribution()



if __name__ == "__main__":
    main()
