#This program builds a soft-margin linear SVM from scratch using the dual formulation.
import numpy as np
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt


#global variables
X = None                    #training points, shape (N, d)
t = None                    #Labels, shape (N,), values +1 or -1
P = None                    #Precomputed matrix P[i, j] = t_i * t_j * K(x_i, x_j)
C = None                    #Soft-margin parameter; None means hard margin
threshold = 1e-5            #anything smaller is treated as zero

#learned values after training
alpha = None                #all optimized alpha values
support_alphas = None       #only non-zero alpha values
support_vectors = None      #the important training points
support_targets = None      #labels of those support vectors
b = None                    #bias term used in classification

#kernel function
#K(x, y) = x dot y
#it measures similarity between two points
def kernel(x, y):
  return np.dot(x, y)

#Polynomial kernel
#p = 2
#def kernel(x, y):
#    return (np.dot(x, y) + 1) ** p

#RBF kernel
#sigma = 0.5
#def kernel(x, y):
#    diff = x - y
#    return np.exp(-np.dot(diff, diff) / (2 * sigma**2))


#build the P matrix
#P[i,j] = t_i * t_j * K(x_i, x_j)
#the objective function is called many times by the optimizer
#so we precompute this once to avoid repeated work
def build_P():
    global P, X, t
    N = len(X)
    P = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            P[i, j] = t[i] * t[j] * kernel(X[i], X[j])

#Objective function
#equation 4: 1/2 * alpha^T P alpha - sum(alpha)
#this is the function the optimizer minimizes,
#it takes one candidate alpha vector and returns its costs
def objective(a):
    return 0.5 * np.dot(a, np.dot(P, a)) - np.sum(a)


#equality constraint function
#equation 10: sum(alpha_i * t_i) = 0
#SciPy will force zerofun(a) == 0
def zerofun(a):
    return np.dot(a, t)

#training function
#this function
# - stores training data
# - builds P
# - sets bounds
# - calls scipy.optimize.minimize
# - extracts support vectors
# - computes b
def train_svm(X_input, t_input, C_value=None):
    global X, t, C, alpha
    global support_alphas, support_vectors, support_targets, b

    X = np.asarray(X_input, dtype=float)
    t = np.asarray(t_input, dtype=float)
    C = C_value

    N = len(X)

    #build P once before optimization
    build_P()

    #initial guess for alpha:
    #start with all zeros
    start = np.zeros(N)

    #Bounds on alpha
    #Hard margin: alpha_i >= 0
    if C is None:
        B = [(0, None) for _ in range(N)]
    else:
        #Soft margin: 0 <= alpha_i <= C
        B = [(0, C) for _ in range(N)]

    #equality constraint dictionary for SciPy:
    #zerofun(alpha) must equal 0
    XC = {"type": "eq", "fun": zerofun}

    #Run the optimizer
    ret = minimize(objective, start, bounds=B, constraints=XC)

    #Check if optimization succeeded
    if not ret["success"]:
        raise RuntimeError(f"Optimization failed: {ret.get('message', 'unknown error')}")
    
    #The learned alpha values are stores under key "x"
    alpha = ret["x"]

    #extract support vectors
    #support vectors are points with alpha_i > 0
    #we use a threshold
    mask = alpha > threshold
    support_alphas = alpha[mask]
    support_vectors = X[mask]
    support_targets = t[mask]

    if len(support_alphas) == 0:
        raise RuntimeError("No support vectors found.")
    
    #Compute b using equation 7
    # b = sum(alpha_i * t_i * K(s, x_i)) - t_s
    # where s is a support vector on the margin.
    # In soft margin:
    # use a support vector with 0 < alpha_i < C if possible.
    if C is None:
        #Hard margin: any support vector can be used
        margin_indices = np.where(mask)[0]
    else:
        #Soft margin: prefer support vectors with 0 < alpha < C
        margin_indices = np.where((alpha > threshold) & (alpha < C - threshold))[0]

        #If none are found exactly (numerical issues)
        #Fall back to any support vector
        if len(margin_indices) == 0:
            margin_indices = np.where(mask)[0]

    #Pick one support vector index
    k = margin_indices[0]
    s = X[k]
    t_s = t[k]

    #Compute the sum
    total = 0.0
    for i in range(N):
        total += alpha[i] * t[i] * kernel(s, X[i])

    b = total - t_s

    return ret

#indicator function
#equation 6: indicator(s) = sum(alpha_i * t_i * K(s, x_i)) - b
#this is the final classifier score:
#if it is positive => class +1
#if it is negative => class -1
#we only sum over support vectors
#because all other alpha values are zero
def indicator(s):
    s = np.asarray(s, dtype=float)

    total = 0.0
    for a_i, t_i, x_i in zip(support_alphas, support_targets, support_vectors):
        total += a_i * t_i * kernel(s, x_i)

    return total - b

def predict(s):
    return 1 if indicator(s) > 0 else -1

#PART 5: GENERATING TEST DATA

np.random.seed(100)

#Class A: two clusters
classA = np.concatenate((
    np.random.randn(10, 2) * 0.2 + [1.5, 0.5], #10 points near (1.5, 0.5)
    np.random.randn(10, 2) * 0.2 + [-1.5, 0.5] #10 points near (-1.5, 0.5)
))

#Class B: one cluster
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

#Combine all points
inputs = np.concatenate((classA, classB))

#Labels: +1 for classA, -1 for classB
targets = np.concatenate((
    np.ones(classA.shape[0]),
    -np.ones(classB.shape[0])
))

#Number of samples
N = inputs.shape[0]

#Shuffle points and labels in the same order
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

#Soft-margin SVM with C = 1.0
ret = train_svm(inputs, targets, C_value=1.0)

print("Optimization success:", ret['success'])
print("\nAll alpha values:")
print(alpha)

print("\nSupport vectors:")
print(support_vectors)

print("\nSupport vector labels:")
print(support_targets)

print("\nSupport alpha values:")
print(support_alphas)

print("\nb value:")
print(b)

#Test a new point
test_point = np.array([0.0, 0.0])
print("\nTest point:", test_point)
print("Indicator:", indicator(test_point))
print("Predicted class:", predict(test_point))



#Plot original classes
plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.', label='Class A (+1)')

plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.', label='Class B (-1)')

#Highlight support vectors with larger black circles
plt.plot([p[0] for p in support_vectors],
         [p[1] for p in support_vectors],
         'ko', markersize=10, fillstyle='none', label='Support vectors')

#Create a grid of points and evaluate the indicator
xgrid = np.linspace(-5, 5, 200)
ygrid = np.linspace(-4, 4, 200)

grid = np.array([[indicator(np.array([x, y]))
                  for x in xgrid]
                 for y in ygrid])

#Plot contour lines:
#-1 and +1 are the margin lines
# 0 is the decision boundary
plt.contour(xgrid, ygrid, grid,
            levels=(-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidths=(1, 3, 1))

#Plot the test point too (optional)
plt.plot(test_point[0], test_point[1], 'gs', markersize=8, label='Test point')

plt.axis('equal')
plt.legend()
plt.title('SVM: data, support vectors, decision boundary, and margins')
plt.savefig('svmplot.pdf')
plt.show()
