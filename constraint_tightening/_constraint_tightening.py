import numpy as np

from scipy import stats


class ConstraintTightening:
    def __init__(self, G_u, g_u, G_x, g_x, risk_factor=0.975):
        self.G_u = np.array(G_u, dtype=float)
        self.g_u = np.array(g_u, dtype=float)
        self.G_x = np.array(G_x, dtype=float)
        self.g_x = np.array(g_x, dtype=float)

        # Initialize pseudo constraints
        self.G_v = self.G_u.copy()
        self.g_v = self.g_u.copy()
        self.G_z = self.G_x.copy()
        self.g_z = self.g_x.copy()

        self.numbr_state_constr = self.G_x.shape[0]
        self.numbr_states = self.G_x.shape[1]

        self.p = risk_factor

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
        interv_min = -50
        interv_max = 50

        x_eval_pdf = np.linspace(interv_min, interv_max, number_eval_points)

        # Iterate over every state constraint
        for idx_c in range(0, self.numbr_state_constr):

            distributions_for_convolution = list()

            # Iterate over every state
            for idx_s in range(0, self.numbr_states):

                coeff_state = self.G_x[idx_c, idx_s]

                # Only consider state if distribution part of (joint) constraint
                if np.abs(coeff_state) > 0.001:

                    # Calculate linearly transformed pdf on interval
                    # Z = aY -> f_z(x) = 1/|a| * f_y(x/a)
                    interval = np.linspace(
                        interv_min/coeff_state, interv_max/coeff_state, number_eval_points)
                    transf_pdf_on_interv = (
                        1/np.abs(coeff_state)) * kde_of_states[idx_s].evaluate(interval)

                    distributions_for_convolution.append(transf_pdf_on_interv)

            # Linear combination of random variables in self.G_x[idx_c] results in convolution of transformed pdf
            # Z = X + Y -> f_z(x) = f_x(x) x f_y(x)

            conv_pdf = distributions_for_convolution[0]

            if len(distributions_for_convolution) > 1:
                for i in range(1, len(distributions_for_convolution)):
                    conv_pdf = np.convolve(
                        conv_pdf, distributions_for_convolution[i], 'same')

            # Now conv_pdf is the pdf resulting from the linear combination of the pdf's of the states on the interval [-10,10]

            # Calculate beta with P(Z>=beta) <= 1-risk_factor
            prob_distr_integr = np.cumsum(
                conv_pdf) * (interv_max-interv_min)/number_eval_points
            idx_upper_bound = np.searchsorted(
                prob_distr_integr, self.p, side='right')-1
            upper_bound = x_eval_pdf[idx_upper_bound] if idx_upper_bound < number_eval_points else 0

            self.g_z[idx_c] = self.g_x[idx_c] - upper_bound

        return self.G_v.copy(), self.g_v.copy(), self.G_z.copy(), self.g_z.copy()

    def tighten_constraints_on_multivariate_kde(self, multi_kde):
        """Tighten constraints based on one multivariate KDE"""

        # The current implementation only allows constraints which involve one or two states

        # Each distribution is always evaluated on the same interval
        number_eval_points = 201
        # interv_min and interv_max have to be chosen symmetrically to 0, e.g. abs(interv_min)==abs(interv_max)
        interv_min = -1.0
        interv_max = 1.0

        x_eval_pdf = np.linspace(interv_min, interv_max, number_eval_points)

        # Iterate over every state constraint
        for idx_c in range(0, self.numbr_state_constr):

            # Store indices of states, which are affected by constraints
            involved_states = list()

            # Iterate over every state
            for idx_s in range(0, self.numbr_states):

                if np.abs(self.G_x[idx_c, idx_s]) > 0.001:
                    involved_states.append(idx_s)

            if len(involved_states) == 1:
                # Tighten constraint which only affects a single state
                marginal_kde = self.det_marginal_distribution(
                    multi_kde, involved_states)

                coeff_state = self.G_x[idx_c, involved_states[0]]

                # Calculate linearly transformed pdf on interval
                # Z = aY -> f_z(x) = 1/|a| * f_y(x/a)
                interval = np.linspace(
                    interv_min/coeff_state, interv_max/coeff_state, number_eval_points)
                transf_pdf_on_interv = (
                    1/np.abs(coeff_state)) * marginal_kde.evaluate(interval)

                # Check, if significant parts of distribution are likely to be outside of interval
                if transf_pdf_on_interv[0] >= 0.00001 or transf_pdf_on_interv[0] >= 0.00001:
                    msg = "Probability of disturbance lying outside of specified interval not neglectable."
                    raise ValueError(msg)

                # Calculate beta with P(Z>=beta) <= 1-risk_factor
                prob_distr_integr = np.cumsum(
                    transf_pdf_on_interv) * (interv_max-interv_min)/number_eval_points
                idx_upper_bound = np.searchsorted(
                    prob_distr_integr, self.p, side='right')-1
                upper_bound = x_eval_pdf[idx_upper_bound] if idx_upper_bound < number_eval_points else 0

                self.g_z[idx_c] = self.g_x[idx_c] - upper_bound

            elif len(involved_states) == 2:
                # Tighten joint constraint
                marginal_kde = self.det_marginal_distribution(
                    multi_kde, involved_states)

                # Get coefficients of involved states (Z = coeff_state_1 * X + coeff_state_2 * y <= c)
                coeff_state_1 = self.G_x[idx_c, involved_states[0]]
                coeff_state_2 = self.G_x[idx_c, involved_states[1]]

                # Calculate linearly transformed pdfs on interval
                # Z_1 = aX -> f_z1(x) = 1/|a| * f_x(x/a)
                interval_1 = np.linspace(
                    interv_min/coeff_state_1, interv_max/coeff_state_1, number_eval_points)
                # Z_2 = bY -> f_z2(x) = 1/|b| * f_y(x/b)
                interval_2 = np.linspace(
                    interv_min/coeff_state_2, interv_max/coeff_state_2, number_eval_points)

                # Calculate matrix to evaluate pdf on
                matrix_to_eval = list()
                for pos_1 in interval_1:
                    for pos_2 in interval_2:
                        matrix_to_eval.append([pos_1, pos_2])
                matrix_to_eval = np.array(matrix_to_eval).transpose()

                pdf_on_matrix = (1/np.abs(coeff_state_1*coeff_state_2)) * \
                    marginal_kde.evaluate(matrix_to_eval)
                # Get pdf realizations in matrix form
                pdf_on_matrix = pdf_on_matrix.reshape(
                    number_eval_points, number_eval_points)

                # Check, if significant parts of distribution are likely to be outside of interval
                if pdf_on_matrix[0,0] >= 0.00001 or pdf_on_matrix[-1,-1] >= 0.00001 or pdf_on_matrix[0,-1] >= 0.00001 or pdf_on_matrix[-1,0] >= 0.00001:
                    msg = "Probability of disturbance lying outside of specified interval not neglectable."
                    raise ValueError(msg)


                # Normalize pdf
                # IMPORTANT: normalizing pdf is only correct, if likelihood of samples lying outside of matrix_to_eval is neglectable
                sum_matrix = np.sum(pdf_on_matrix)
                pdf_on_matrix = pdf_on_matrix/sum_matrix

                # Traverse matrix from highest disturbance to zero disturbance and sum up probabilities of the corresponding disturbance realizations
                # Start with lower right corner (corresponds to highest disturbance)
                # Stop at diagonal of matrix (round < number_eval_points), because then sum of disturbance is 0
                # Going further would lead to constraint relaxing
                sum_probability = 0
                round = 0
                while sum_probability < 1 - self.p and round < number_eval_points:
                    for i in range(0, round+1):
                        idx_1 = number_eval_points-1-i
                        idx_2 = number_eval_points-1-round+i
                        sum_probability += pdf_on_matrix[idx_1, idx_2]

                    round += 1

                # P(Z = Z_1 + Z_2 <= y) >= p
                upper_bound = ((number_eval_points-round)/number_eval_points)*(interv_max)*2

                self.g_z[idx_c] = self.g_x[idx_c] - upper_bound

            elif len(involved_states) > 2:
                msg = "Joint constraints are not allowed to involve more than two states currently."
                raise ValueError(msg)

        return self.G_v.copy(), self.g_v.copy(), self.G_z.copy(), self.g_z.copy()

    def det_marginal_distribution(self, kde, dimensions):
        """"Return marginal distribution of kde for dimensions
        This function is largely taken from the current (28.09.2022) KDE implementation of the main branch of scipy
        https://github.com/scipy/scipy/blob/dd153ceab933e74ab33c9391445dc8686c28479a/scipy/stats/_kde.py#L629
        """

        dims = np.atleast_1d(dimensions)

        if not np.issubdtype(dims.dtype, np.integer):
            msg = ("Elements of `dimensions` must be integers - the indices "
                   "of the marginal variables being retained.")
            raise ValueError(msg)

        n = len(kde.dataset)  # number of dimensions
        original_dims = dims.copy()

        dims[dims < 0] = n + dims[dims < 0]

        if len(np.unique(dims)) != len(dims):
            msg = ("All elements of `dimensions` must be unique.")
            raise ValueError(msg)

        i_invalid = (dims < 0) | (dims >= n)
        if np.any(i_invalid):
            msg = (f"Dimensions {original_dims[i_invalid]} are invalid "
                   f"for a distribution in {n} dimensions.")
            raise ValueError(msg)

        dataset = kde.dataset[dims]
        weights = kde.weights

        return stats.gaussian_kde(dataset, bw_method=kde.covariance_factor(),
                                  weights=weights)
