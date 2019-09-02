# -*- coding: utf-8 -*-
"""
vertigo.py
Functions and objects for the (local) implementation of VERTIGO.
See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4901373/

Created on Wed Apr 24 10:44:44 2019
@author: Arturo Moncada-Torres
arturomoncadatorres@gmail.com
"""

#%% Preliminaries
import pandas as pd
import numpy as np
import numpy.matlib as npm
import statsmodels as sm
import warnings


#%%
class vertigo:
    """
    Class that implements VERTIGO. It is implemented using Fixed-Hessian 
    Newton method.

    Attributes
    ----------
    lambda_: float
        Regularization value.
    n_iter: float
        Max number of iterations.
    alpha_init: float
        Initial value for alpha.
    tol: float
        Tolerance value to claim convergence of the alpha coefficients.
    verbose: boolean
        If true, status information will be printed in console.

    Methods
    -------
    _sigmoid(x)
        Sigmoid function.
    _is_diagonally_dominant(x)
        Check if matrix is diagonally dominant
    _is_positive_definite(x)
        Check if matrix is positive definite
    _compute_beta
        Compute beta coefficients according to Eq. 5
    fit
        Fit logistic regression model
    predict
        Obtain predictions, TODO
        
    References
    -----
    Li, Yong, et al. "Vertical grid logistic regression (vertigo)." Journal of 
    the American Medical Informatics Association 23.3 (2015): 570-579.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4901373/
    """
    
    def __init__(self, lambda_=1000, n_iter=500, alpha_init=0.5, tol=1e-8, verbose=True):
        """
        Initialization.
        
        Parameters
        ----------
        lambda_: float
            Regularization value.
        n_iter: float
            Max number of iterations.
        alpha_init: float
            Initial value for alpha.
        tol: float
            Tolerance value to claim convergence of the alpha coefficients.
        verbose: boolean
            If true, status information will be printed in console.
        
        Returns
        -------
        None
        """

        self.lambda_ = lambda_
        self.n_iter = n_iter
        self.alpha_init = alpha_init
        self.tol = tol
        self.verbose = verbose


    def _sigmoid(self, x):
        """
        Simple sigmoid function. Bounds values between 0 and 1.
        
        Parameters
        ----------
        x: float
            Array of values to be evaluated.
        
        Returns
        -------
        float
            Corresponding sigmoid values.
            
        References
        ----------
        https://www.computing.dcu.ie/~humphrys/Notes/Neural/sigmoid.html
        """        
        return 1/(1 + np.exp(-x))


    def _is_diagonally_dominant(self, X):
        """
        Check if a squared matrix is diagonally dominant.
        
        Parameters
        ----------
        X: numpy matrix.
            Matrix to check.
        
        Returns
        -------
        boolean
            A boolean defining if input matrix is diagonally dominant (True)
            or not (False)
            
        References
        ----------
        https://en.wikipedia.org/wiki/Diagonally_dominant_matrix
        """
        # Find diagonal coefficients
        D = np.diag(np.abs(X)) 
        
        # Find row sum without diagonal
        S = np.sum(np.abs(X), axis=1) - D
        
        # Check for diagonal dominance.
        if np.all(D > S):
            return True
        else:
            return False
    
    
    def _is_positive_definite(self, X):
        """
        Check if a if matrix is positive definite.
        
        Parameters
        ----------
        X: numpy matrix.
            Matrix to check.
        
        Returns
        -------
        boolean
            A boolean defining if input matrix is positive definite (True)
            or not (False)
            
        References
        ----------
        https://en.wikipedia.org/wiki/Definiteness_of_a_matrix
        """
        return np.all(np.linalg.eigvals(X) > 0)


    def _compute_beta(self, X, Y, lambda_1, alpha):
        """
        Compute beta coefficient according to Eq. 5
        
        Parameters
        ----------
        X: numpy matrix
            Feature vector of one party. It has dimension M x N, where M is 
            the number of patients (the same for all parties) and N is the 
            number of features.
        Y: numpy array
            Response vector. It has M x 1 dimensions.
        lambda_1: float
            Regularization term.
        alpha: numpy array
            Alpha coefficients to be used. It has dimension N x 1.
        
        Returns
        -------
        beta: numpy array
            Vector of beta coefficients. 
        """
        # Number of features.
        N = X.shape[1]

        # Actual computation.
        beta = lambda_1 * np.sum((npm.repmat(alpha, 1, N) * 
                                  npm.repmat(Y, 1, N) *
                                  X), 0)
        return beta
        

    def fit(self, X, Y):
        """
        Fit VERTIGO. Obtains beta (and alpha) coefficients.
        
        Parameters
        ----------
        X: list of dataframes or numpy matrices.
            Input. Each element i corresponds to that party's X (feature) 
            matrix. Each matrix has shape (M, N_i), where M is the number
            of patients (the same for all parties) and N_i is the number of
            features for party i. Data does not need to have a constant term
            (for the model intercept), since it is added automatically.
            Datasets of all patients have to be aligned. If any element is a 
            data frame, it will be converted to a numpy matrix.
        Y: numpy array. Shape (M, 1)
            Response vector. All parties have access
            to it.
        
        Returns
        -------
        None
        """


        #%%
        # 0. Preliminaries.
        
        # If necessary, convert data frames to numpy matrices.
        for idx, party in enumerate(X):
            if isinstance(party, pd.DataFrame):
                X[idx] = party.to_numpy()
                if self.verbose:
                    print("Converted party {0:d} from data frame to matrix".format(idx+1))
                    
        if isinstance(Y, pd.DataFrame):
            Y = Y.to_numpy()
            if self.verbose:
                print("Converted Y from data frame to matrix")
        
        # Add column of 1s to serve as constant term in the computation of the
        # residual coefficient.
        for idx, party in enumerate(X):
            X[idx] = sm.tools.add_constant(party, prepend=False)
            if self.verbose:
                print("Added constant term to data from party {0:d}".format(idx+1))

        LAMBDA_1 = 1/self.lambda_
        M = X[0].shape[0] # Number of patients (same for all parties).
        Y_diag = np.diag(Y.flatten()) # Diagonal version of Y
        ALPHA_INIT = self.alpha_init * np.ones((M, 1)) # Initialize alpha vector


        #%%
        # 1. LOCAL
        
        # 1.1 Compute local gram matrix (i.e., linear kernel matrix).
        if self.verbose:
            print("Computing local gram matrices...", end='', flush=True)
        
        calculate_gram_matrix = lambda x: np.matmul(x, x.T)
        K_all = list(map(calculate_gram_matrix, X))
        
        if self.verbose:
            print("\tDONE!")
        
        
        # 2. SERVER
        
        # 2.1 Calculate global gram matrix K (Eq. 3)
        if self.verbose:
            print("Computing global gram matrix...", end='', flush=True)
        K = np.sum(K_all, axis=0)            
        if self.verbose:
            print("\t\tDONE!")
            
            

        # 2.2 Calculate fixed Hessian matrix H_wave (Eq. 10)
        # Depending on the data, this can take a while. In my experiences,
        # for a 5000 patients dataset about 6-7 min.
        
        # c must be set to a value that makes H_wave diagonally dominant.
        # We will initialize it as "the maximum of the elements in the 
        # original H" (p. 573). We will increase c by a factor in steps of 0.5
        # until the condition of diagonally dominance is satisfied.
        not_diagonally_dominant = True
        c_factor = 1
        
        while not_diagonally_dominant:
            
            if self.verbose:
                print("Obtaining H_wave with c_factor = {0:.1f}...".format(c_factor))
            
            H = LAMBDA_1 * np.matmul(np.matmul(Y_diag, K), Y_diag) + np.diag(1/(ALPHA_INIT*(1-ALPHA_INIT)))
            c =  c_factor * H.max()
            I = np.identity(len(Y_diag))
        
            H_wave = LAMBDA_1 * np.matmul(np.matmul(Y_diag, K), Y_diag) + c*I
            not_diagonally_dominant = not self._is_diagonally_dominant(H_wave)
            c_factor += 0.5
            
            if self.verbose:
                if not_diagonally_dominant:
                    print("\tH_wave is NOT diagonally dominant")
                else:
                    print("\tH_wave is diagonally dominant")
                    
            # Uncomment break to not make sure that H_wave is diagonally dominant.
            break
        
        if self.verbose:
            if self._is_positive_definite(H_wave):
                print("\tH_wave is positive definite")
            else:
                print("\tH_wave is NOT positive definite")
                    
        # 2.3 Calculate H^-1
        # This is fast (~5 s)
        if self.verbose:
            print("Obtaining H_wave_1...", end='', flush=True)
        H_wave_1 = np.linalg.inv(H_wave) 
        if self.verbose:
            print("\t\t\tDONE!")
            

        # 3. LOCAL + SERVER
        if self.verbose:
            print("Computing alpha_star...")
            
        # To analyze alpha coefficients
        self.alpha_all = np.zeros((Y.shape[0], self.n_iter))
        
        for s in range(0, self.n_iter):    
            
            if s==0:
                alpha_old = ALPHA_INIT 
            else:
                alpha_old = alpha
        
            self.alpha_all[:, s] = alpha_old.T
            
            if self.verbose:
                print("\ts = {0:d}, alpha[0] = {1:3.5f}".format(s+1, alpha_old[0,0]))
        
            # 3.1 LOCAL
            # Compute betas and E.
            E_all = list()
            for X_party in X:
                beta = self._compute_beta(X_party, Y, LAMBDA_1, alpha_old)
                
                # In order for the matrix multiplications to work, Y needs to 
                # be transposed (which is not stated in the paper).
                E_all.append(Y.T * (np.matmul(beta, X_party.T)))
        
            # 3.2 SERVER
            E = np.sum(E_all, axis=0)   
            
            # Catch incorrect cases (logarithm of a negative number)
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    J = E.T + np.log10(alpha_old/(1-alpha_old))
                except Warning as e:
                    print('WARNING:', e)
            
            # 3.3 SERVER
            # Update alpha.
            alpha = alpha_old - np.matmul(H_wave_1, J)
                
            
            # Bound alpha (0 < alpha < 1)
            # This isn't defined in the original paper, but is done to avoid
            # problems when computing J (the logarithm of a negative number is
            # undefined).
            alpha = self._sigmoid(alpha)
        
            # Check for convergence.
            if max(abs(alpha - alpha_old)) < self.tol:
                # Trim alpha_all (matrix for visualization)
                self.alpha_all = np.delete(self.alpha_all, np.s_[s:], axis=1)
                if self.verbose:
                    print("Convergence reached at iteration {0:d}".format(s+1))
                break
            
        # 4. Final alpha value.
        self.alpha_star = alpha
        

        # 5. Obtain beta coefficients.
        # This step isn't defined in Fig. 4, but it is the logical next step.
        # This is done locally.
        if self.verbose:
            print("Computing global beta coefficients...", end='', flush=True)
        
        self.coefficients = list()
        for X_party in X:
            self.coefficients.append(self._compute_beta(X_party, Y, LAMBDA_1, self.alpha_star))

        # Pack into proper format.
        self.beta_coef_ = list()
        for coefficient_set in self.coefficients:
            self.beta_coef_.extend(coefficient_set[0:-1])
        self.intercept_ = self.coefficients[0][-1]

        if self.verbose:
            print("\t\tDONE!")


    def predict(self, Z):
        """
        Obtain predictions.
        TODO: Unfinished. Untested.
        
        Parameters
        ----------
        Z: list of numpy matrices.
            Input. Each element i corresponds to that party's X (feature) 
            matrix. Matrices have M x N_i dimensions, where M is the number
            of patients (the same for all parties) and N_i is the number of
            features for party i. Datasets of all patients have to be aligned.
        
        Returns
        -------
        XXX
        """
        # SERVER
        if self.verbose:
            print("Generating predictions...", end='', flush=True)
        
        F_all = list()
        for idx, Z_party in enumerate(Z):
            F_all.append(np.matmul(self.coefficients[idx][0:-1], Z_party.T))
            if self.verbose:
                print("Computed F of party {0:d}".format(idx+1))

        F = np.sum(F_all, axis=0) 
        self.y_prob = self._sigmoid(F)
        if self.verbose:
            print("\t\tDONE!")
            