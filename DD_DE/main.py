from DD_DE import lti_system
from DD_DE import disturbance_estimator
from DD_DE import data_driven_predictor


def main():
    print("Hello World!")
    my_system = lti_system.LTISystem(0,1,2)
    print(my_system.x)
    my_system.next_step(1)
    print(my_system.x)
    my_system.next_step(2)
    print(my_system.x)



if __name__ == "__main__":
    main()
