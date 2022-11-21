LTI-System Matrix is faulty:

def next_step(self, u, add_disturbance=True,k=None,add_measurement_noise=False):
        self.u = np.array(u).reshape(-1, 1)

        if add_disturbance and k!=None:
            self.x = self.A @ self.x + self.B @ self.u + self.dist_seq[:,k].reshape(-1,1) + np.array([[0,0,0.03,0],[0,0,0,0.03],[0,0,0,0],[0,0,0,0]])@self.x
        elif add_measurement_noise:
            self.x = self.A @ self.x + self.B @ self.u + np.random.normal(loc=0,scale=0.001,size=4).reshape(-1,1)
        else:
            self.x = self.A @ self.x + self.B @ self.u

        # print(self.x)
        self.k += 1
        return self.x
