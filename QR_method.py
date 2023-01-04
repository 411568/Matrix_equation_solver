import numpy as np

#householder decomposition to Q and R matrices
def qr(A):
    n = A.shape[0]
    Q = np.eye(n)

    for i in range(n-1):
        H = np.eye(n)
        H[i:, i:] = make_householder(A[i:, i])
        #Q = QH
        Q = Q @ H
        #R = HR
        A = H @ A
        
    return Q, A
 
#returns a householder matrix
def make_householder(A):
    u = A / (A[0] + np.copysign(np.linalg.norm(A), A[0]))
    u[0] = 1
    H = np.eye(A.shape[0])

    H = H - (2 / np.dot(u, u)) * np.transpose(u[np.newaxis]) @ u[np.newaxis]

    return H

#used to solve the last matrix equation where R is an upper triangular matrix
def back_substitution(R, y):
    n = R.shape[0]
    x = np.zeros(n)

    x[n-1] = y[n-1] / R[n-1][n-1]

    for k in range(n-2, -1, -1):
        s = 0
        for p in range(n-1, -1, -1):
            s = s + (R[k][p] * x[p])
            
        x[k] = (y[k] - s) / R[k][k]

    return x


#function that solves the equation QRx = B
def solve_qr(Q, R, B):
    #QRx = B
    Q = np.transpose(Q)
    #R * x = Qt * B
    Y = Q @ B
    X = back_substitution(R, Y)

    return X



#define the A matrix
A = np.array([[1,1,1],
              [1,2,3],
              [1.5,2,4]])

#define the B matrix
B = np.array([[1],
              [1],
              [1]])

#solve the equation
Q, R = qr(A)
X = solve_qr(Q, R, B)

print("A :")
print(A)
print("\nB :")
print(B)
print("\nR : ")
print(R)
print("\nQ : ")
print(Q)
print("\nX - solution : ")
print(np.transpose(X[np.newaxis]))
