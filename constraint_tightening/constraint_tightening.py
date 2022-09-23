import numpy as np


class ConstraintTightening:
    def __init__(self, G_u, g_u, G_x, g_x, risk_factor=0.95):
        self.G_u = np.array(G_u)
        self.g_u = np.array(g_u)
        self.G_x = np.array(G_x)
        self.g_x = np.array(g_x)

        # Initialize pseudo constraints
        self.G_v = self.G_u.copy()
        self.g_v = self.g_u.copy()
        self.G_z = self.G_x.copy()
        self.g_z = self.g_x.copy()

        self.numbr_state_constr = self.G_x.shape[0]
        self.numbr_states = self.G_x.shape[1]

        self.p = risk_factor

    def tighten_constraints(self):
        x = 1

    def get_tightened_constraints(self):
        return self.G_v, self.g_v, self.G_z, self.g_z

    def tighten_constraints_on_interv(self, dist_interval):
        """Tighten constraints based on disturbance intervals"""

        # Todo: Create more flexible solution
        # Currently solution is hardcoded for systems with no joint constraints (Every state has it's own constraints)
        # No joint constraints means, that G_x is of the form G_x = [[1,0,0],[0,1,0,[1,0,0],[0,0,1]]]

        for i in range(0, self.numbr_state_constr):
            # Calculate dist interval for state, which corresponds to constraint
            # --> Only works, if there are no joint constraints
            interval = np.abs(self.G_x[i, :]) @ dist_interval
            if (np.sum(self.G_x[i]) > 0):
                self.g_z[i] = self.g_x[i]+interval[0]
            else:
                self.g_z[i] = self.g_x[i]+interval[1]

        return self.G_v.copy(), self.g_v.copy(), self.G_z.copy(), self.g_z.copy()

    def tighten_constraints_on_indep_kde(self, kde_of_states):
        """Tighten constraints based on independent disturbance distributions from every state"""

        # Each distribution is always evaluated on the same interval
        number_eval_points = 1000
        interv_min = -10
        interv_max = 10

        x_eval_pdf = np.linspace(interv_min, interv_max, number_eval_points)

        
        # Iterate over every state constraint
        for idx_c in range(0, self.numbr_state_constr):
            
            distributions_for_convolution = list()
            
            # Iterate over every state
            for idx_s in range(0,self.numbr_states):

                coeff_state = self.G_x[idx_c,idx_s]
                
                # Only consider state if distribution part of (joint) constraint
                if np.abs(coeff_state) > 0.001:
                    
                    # Calculate linearly transformed pdf on interval
                    # Z = aY -> f_z(x) = 1/|a| * f_y(x/a)
                    interval = np.linspace(interv_min/coeff_state,interv_max/coeff_state,number_eval_points)
                    transf_pdf_on_interv = (1/np.abs(coeff_state)) * kde_of_states[idx_s].evaluate(interval)
                    
                    distributions_for_convolution.append(transf_pdf_on_interv)

            # Linear combination of random variables in self.G_x[idx_c] results in convolution of transformed pdf
            # Z = X + Y -> f_z(x) = f_x(x) x f_y(x)

            conv_pdf = distributions_for_convolution[0]
            
            if len(distributions_for_convolution) > 1:
                for i in range(1,len(distributions_for_convolution)):
                    conv_pdf = np.convolve(conv_pdf,distributions_for_convolution[i],'same')

            # Now conv_pdf is the pdf resulting from the linear combination of the pdf's of the states on the interval [-10,10]

            # Calculate beta with P(Z>=beta) <= 1-risk_factor
            prob_distr_integr = np.cumsum(conv_pdf) * (interv_max-interv_min)/number_eval_points
            idx_upper_bound = np.searchsorted(prob_distr_integr, self.p, side='right')-1
            upper_bound = x_eval_pdf[idx_upper_bound] if idx_upper_bound < number_eval_points else 0

            self.g_z[idx_c] = self.g_x[idx_c] - upper_bound

        return self.G_v.copy(), self.g_v.copy(), self.G_z.copy(), self.g_z.copy()




            

