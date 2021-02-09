import numpy as np
import warnings
from cvxopt import matrix, solvers

from ..base.optimizers import Optimizer
from ..kernels import Kernel, Linear

from ..utils.data_utils import add_cons
from ..utils.initialize_utils import set_X, set_y
from ..utils.decorator_utils import implementation


EPS = np.finfo(np.float64).eps            # Machine epislon: 2.220446049250313e-16
solvers.options['show_progress'] = False  # turn off QP-solver's iteration output
solvers.options['abstol'] = EPS           # Set the tolerance for QP-solver, since lagrange
solvers.options['reltol'] = EPS           #   multipliers are to be compared with zero to
solvers.options['feastol'] = EPS          #   get support vectors.
TOLERANCE = 1e4 * EPS


class PrimalQpSolver(Optimizer):
    def __init__(self, kernel=None, soft_margin_penalty=None, tube_width=None):
        self.kernel = kernel
        self.soft_margin_penalty = soft_margin_penalty  # soft_margin_penalty != None => Soft Margin SVM
        self.tube_width = tube_width                    # tube_width != None => SVR

    def execute(self, x_train, y_train):
        if self.kernel is not None:
            warnings.warn("You are using primal Qp solver. Since some kernels have ill-implemented transformation function, use kernel carefully.")

        x = set_X(self.kernel.transform(x_train), add_constant_terms=False)
        y = set_y(y_train)

        soft_margin_penalty = self.soft_margin_penalty
        tube_width = self.tube_width

        if soft_margin_penalty is None: 
            return self.primal_qp_hard_margin_svm(x, y)
        elif tube_width is None:
            return self.primal_qp_soft_margin_svm(x, y, soft_margin_penalty)
        else:
            return self.primal_qp_svr(x, y, soft_margin_penalty, tube_width)


    @implementation(compile=None)
    def primal_qp_hard_margin_svm(x, y):
        N = x.shape[0]  # number of data
        d = x.shape[1]  # dimension
        
        # Formulation of the primal problem for hard-margin SVM
        Q = matrix(np.block([[0,               np.zeros((1, d)) ],
                            [np.zeros((d, 1)), np.identity(d)   ]
            ]))
        p = matrix(np.zeros(d+1))
        a_t = matrix(np.diag(y) @ np.block([np.ones((N, 1)), x]))
        c = matrix(np.ones((N, 1)))

        # Solve primal problem using cvxopt QP solver
        sol = solvers.qp(P=Q, q=p, G=-a_t, h=-c)
        b = np.array(sol['x']).flatten()[0]         # bias term
        w = np.array(sol['x']).flatten()[1:d+1]     # weight
        return b, w


    @implementation(compile=None)
    def primal_qp_soft_margin_svm(x, y, soft_margin_penalty):
        N = x.shape[0]  # number of data
        d = x.shape[1]  # dimension
        C = soft_margin_penalty

        # Formulation of the primal problem for soft-margin SVM 
        Q = matrix(np.block([[0,               np.zeros((1, d)), np.zeros((1, N))], 
                            [np.zeros((d, 1)), np.identity(d),   np.zeros((d, N))], 
                            [np.zeros((N, 1)), np.zeros((N, d)), np.zeros((N, N))]  
            ]))
        p = matrix(np.vstack([np.zeros((d+1, 1)), C*np.ones((N, 1))]))
        a_t = matrix(np.block([[np.diag(y) @ add_cons(x),   np.identity(N)],
                               [np.zeros((N, d+1)),         np.identity(N)]
            ]))
        c = matrix(np.vstack([np.ones((N, 1)), np.zeros((N, 1))]))
        
        # Solve primal problem using cvxopt QP solver
        sol = solvers.qp(P=Q, q=p, G=-a_t, h=-c)
        b = np.array(sol['x']).flatten()[0]         # bias term
        w = np.array(sol['x']).flatten()[1:d+1]     # weight
        return b, w


    @implementation(compile=None)
    def primal_qp_svr(x, y, soft_margin_penalty, tube_width):
        print(x.shape)
        N = x.shape[0]  # number of data
        d = x.shape[1]  # dimension
        C = soft_margin_penalty
        epsilon = tube_width

        # Formulation of the primal problem for SVR: 
        #   QP with d+1+2N variables, 2N+2N constraints
        Q = matrix(np.block([[0,                 np.zeros((1, d)),   np.zeros((1, 2*N))], 
                            [np.zeros((d, 1)),   np.identity(d),     np.zeros((d, 2*N))], 
                            [np.zeros((2*N, 1)), np.zeros((2*N, d)), np.zeros((2*N, 2*N))]  
            ]))
        p = matrix(np.hstack((np.zeros(d+1), C*np.ones(2*N))))
        a_t = matrix(np.block([[np.ones((N, 1)),  x,                np.identity(N),  np.zeros((N, N)) ],
                               [-np.ones((N, 1)), -x,               np.zeros((N,N)), np.identity(N)   ],
                               [np.zeros((N, 1)), np.zeros((N, d)), np.identity(N),  np.zeros((N,N))  ],
                               [np.zeros((N, 1)), np.zeros((N, d)), np.zeros((N,N)), np.identity(N)   ] 

            ]))
        c = matrix(np.hstack((y-epsilon, -y-epsilon, np.zeros(2*N))))

        # Solve primal problem using cvxopt QP solver
        sol = solvers.qp(P=Q, q=p, G=-a_t, h=-c)
        b = np.array(sol['x']).flatten()[0]         # bias term
        w = np.array(sol['x']).flatten()[1:d+1]     # weight
        print(d)
        print(sol['x'])
        return b, w




class DualQpSolver(Optimizer):
    def __init__(self, kernel=None, soft_margin_penalty=None, tube_width=None):
        self.kernel = kernel
        self.soft_margin_penalty = soft_margin_penalty
        self.tube_width = tube_width


    def execute(self, x_train, y_train):
        x = set_X(x_train, add_constant_terms=False)
        y = set_y(y_train)
        
        K = self.kernel.inner_product         # Kernel function (inner_product)
        soft_margin_penalty = self.soft_margin_penalty
        tube_width = self.tube_width

        
        if soft_margin_penalty is None:
            return self.dual_qp_hard_margin_svm(x, y, K)
        elif tube_width is None:
            return self.dual_qp_soft_margin_svm(x, y, K, soft_margin_penalty)
        else:
            return self.dual_qp_svr(x, y, K, soft_margin_penalty, tube_width)


    @implementation(compile=None)
    def dual_qp_hard_margin_svm(x, y, K):
        N = x.shape[0]

        # Formulation of the dual problem for hard-margin SVM
        Q = matrix((np.diag(y) @ K(x, x) @ np.diag(y)))
        p = matrix(-np.ones((N, 1)))
        a_n = matrix(np.identity(N))
        c_n = matrix(np.zeros((N, 1)))
        a = matrix(y, (1, N), tc='d')   # 'd' is the type-code for "double" in cvxopt
        c = matrix(0, tc='d')

        # Solve dual problem using cvxopt QP solver
        sol = solvers.qp(P=Q, q=p, G=-a_n, h=c_n, A=a, b=c)
        alpha = np.asarray(sol['x']).flatten()   # Lagrange multipliers

        # Find support vectors(SV) 
        sv_i = np.flatnonzero(alpha > (0 + TOLERANCE)) # sv indice
        sv_x = x[sv_i, :]
        sv_y = y[sv_i]
        support_vector = {"x":sv_x, "y":sv_y, "i":sv_i}

        # Evaluate b(constant term) from Complementary Slackness condition
        # Take first SV for calculation
        b = sv_y[0] - (alpha[sv_i] * sv_y) @ K(sv_x, sv_x[0])
        
        return b, alpha, support_vector

    
    @implementation(compile=None)
    def dual_qp_soft_margin_svm(x, y, K, soft_margin_penalty):
        N = x.shape[0]
        C = soft_margin_penalty

        # Formulation of the dual problem for soft-margin SVM
        Q = matrix((np.diag(y) @ K(x, x) @ np.diag(y)))
        p = matrix(-np.ones((N, 1)))
        a_n = matrix(np.vstack((np.identity(N), (-1) * np.identity(N))))
        c_n = matrix(np.vstack((np.zeros((N, 1)), C * np.ones((N,1)))))
        a = matrix(y, (1, N), tc='d')
        c = matrix(0, tc='d')

        # Solve dual problem using cvxopt QP solver
        sol = solvers.qp(P=Q, q=p, G=-a_n, h=c_n, A=a, b=c)
        alpha = np.asarray(sol['x']).flatten() # Lagrange multipliers

        # Find support vector(SV)
        #   <free support vectors>
        fsv_i = np.flatnonzero(np.logical_and(alpha > (0+TOLERANCE), alpha < (C-TOLERANCE))) # free-sv indice
        fsv_x, fsv_y = (x[fsv_i, :], y[fsv_i])
        fsv = {"x":fsv_x, "y":fsv_y, "i":fsv_i}  
        
        #   <bounded support vectors>
        bsv_i = np.flatnonzero(alpha >= (C-TOLERANCE))  # bounded-sv indice: if alpha>(C-TOLERANCE), regard alpha==C 
        bsv_x, bsv_y = (x[bsv_i, :], y[bsv_i])
        bsv = {"x":bsv_x, "y":bsv_y, "i":bsv_i}

        #   <All>
        sv_i = np.hstack((fsv_i, bsv_i))
        sv_x = np.vstack((fsv_x, bsv_x))
        sv_y = np.hstack((fsv_y, bsv_y))
        support_vector = {"free": fsv, "bounded": bsv}

        # Evaluate b(constant term) from Complementary Slackness condition
        if len(fsv_i) > 0:
            # (1) Free support vectors exist
            b = fsv_y[0] - (alpha[sv_i] * sv_y) @ K(sv_x, fsv_x[0]) 
        else:
            # (2) No free support vectors...
            # assign the value of midpoint from the possible range to b 
            # Ref: https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf (p.12, p.14)
            y_grad_f = y[:, None] * (Q @ alpha + p) 
            I_up = np.logical_or(np.logical_and(y==1, alpha<C), np.logical_and(y==-1, alpha>(0+TOLERANCE)))
            I_low = np.logical_or(np.logical_and(y==-1, alpha<C), np.logical_and(y==1, alpha>(0+TOLERANCE)))
            M = np.min(-y_grad_f[I_low])
            m = np.max(-y_grad_f[I_up])
            b = (M + m)/2
        
        return b, alpha, support_vector


    @implementation(compile=None)
    def dual_qp_svr(x, y, K, soft_margin_penalty, tube_width):
        N = x.shape[0]
        C = soft_margin_penalty
        epsilon = tube_width

        # Formulation of the dual problem for SVR
        k = K(x, x)
        Q = matrix(np.block([[k,  -k ],
                            [-k,   k ]]))
        p = matrix(np.hstack((epsilon-y, epsilon+y)))
        a_n = matrix(np.vstack((np.identity(2*N), (-1) * np.identity(2*N))))
        c_n = matrix(np.vstack((np.zeros((2*N, 1)), C * np.ones((2*N, 1)))))
        a = matrix(np.hstack((np.ones(N), (-1)*np.ones(N))), (1, 2*N), tc='d')
        c = matrix(0, tc='d')

        # Solve dual problem using cvxopt QP solver
        sol = solvers.qp(P=Q, q=p, G=-a_n, h=c_n, A=a, b=c)
        alpha = np.asarray(sol['x']).flatten()
  
        alpha_upper, alpha_lower = (alpha[0:N], alpha[N:2*N])
        beta = alpha_upper - alpha_lower

        # Find support vector(SV)
        #   <free support vectors>
        fsv_i = np.flatnonzero(np.logical_and(np.abs(beta) > (0+TOLERANCE),                 # sv indice: beta != 0: on or outside tube
                                              np.logical_and(alpha_upper < (C-TOLERANCE),
                                                             alpha_lower < (C-TOLERANCE)))) # free-sv indice
        fsv_x, fsv_y = (x[fsv_i, :], y[fsv_i])
        fsv = {"x":fsv_x, "y":fsv_y, "i":fsv_i}
        
        #   <bounded support vectors>
        bsv_i = np.flatnonzero(np.logical_and(np.abs(beta) > (0+TOLERANCE),                  # sv indice: beta != 0: on or outside tube
                                              np.logical_or(alpha_upper >= (C-TOLERANCE),
                                                            alpha_lower >= (C-TOLERANCE))))  # bounded-sv indice
        bsv_x, bsv_y = (x[bsv_i, :], y[bsv_i])
        bsv = {"x":bsv_x, "y":bsv_y, "i":bsv_i}
        
        #   <All support vectors>
        sv_i = np.hstack((fsv_i, bsv_i))
        sv_x = np.vstack((fsv_x, bsv_x))
        sv_y = np.hstack((fsv_y, bsv_y))
        support_vector = {"free": fsv, "bounded": bsv}

        # Evaluate b(constant term) from Complementary Slackness condition
        if len(fsv_i) > 0:
            # (1) Free support vectors exist...
            # Take first free support vector to evaluate b
            if beta[fsv_i[0]] > 0:
                b = fsv_y[0] - tube_width - beta[sv_i] @ K(sv_x, fsv_x[0])
            else:
                b = fsv_y[0] + tube_width - beta[sv_i] @ K(sv_x, fsv_x[0])
        else:
            # (2) No free support vectors...
            # assign the value of midpoint from the possible range to b 
            # Ref: https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf (p.12, p.14)
            y_grad_f = y[:, None] * (Q @ alpha + p) 
            I_up = np.logical_or(np.logical_and(y==1, alpha<C), np.logical_and(y==-1, alpha>(0+TOLERANCE)))
            I_low = np.logical_or(np.logical_and(y==-1, alpha<C), np.logical_and(y==1, alpha>(0+TOLERANCE)))
            M = np.min(-y_grad_f[I_low])
            m = np.max(-y_grad_f[I_up])
            b = (M + m)/2
            
        return b, beta, support_vector