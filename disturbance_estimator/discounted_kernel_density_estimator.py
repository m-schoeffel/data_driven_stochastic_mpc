class DiscountedKDE:
    def __init__(self):

        # Store tuples of index k and delta x (difference between prediction and actual state)
        self.list_delta_x = list()

    def add_delta_x(self, index_k, delta_x):
        self.list_delta_x.append((index_k, delta_x))

    def plot_distribution(self):
        # Todo: Distribution should be plottet here
        # Right now self.delta will be displayed

        print(self.list_delta_x)
