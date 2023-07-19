import numpy as np
from scipy.linalg import rq

x = np.array([757, 758, 758, 759, 1190, 329, 1204, 340])
y = np.array([213, 415, 686, 966, 172, 1041, 850, 159])

X = np.array([0, 0, 0, 0, 7, 0, 7, 0])
Y = np.array([0, 3, 7, 11, 1, 11, 9, 1])
Z = np.array([0, 0, 0, 0, 0, 7, 0, 7])

# A = np.zeros((2*len(x), 12))
A = []
for i in range(len(x)):
    A.append([0, 0, 0, 0, -X[i], -Y[i], -Z[i], -1, y[i]*X[i], y[i]*Y[i], y[i]*Z[i], y[i] ])
    A.append([X[i], Y[i], Z[i], 1, 0, 0, 0, 0, -x[i]*X[i], -x[i]*Y[i], -x[i]*Z[i], -x[i]])

A = np.array(A)
# print(A)

U, S, Vt = np.linalg.svd(A)

P = Vt[-1, :].reshape((3, 4))

M = P[:, :3]

print("Projection Matrix = \n", P)
print("\n")

R, Q = rq(M)

# if np.linalg.det(R) < 0:
#     R[:, 2] *= -1
#     Q *= -1

R = R / R[2,2]

inv_k = np.linalg.inv(R)

T = inv_k @ P

translation = T[:, 3]

print("Intrinsic Matrix = \n", R)
print("\n")
print("Rotation Matrix = \n", Q)
print("\n")
print("Translation Matrix  = \n", translation)
print("\n")

reprojecton_error = []

for i in range(len(x)):

    world_coordinates = np.array([X[i], Y[i], Z[i], 1])

    new_image_coordinates = P @ world_coordinates

    new_image_coordinates_nonhomo = np.array([(new_image_coordinates[0]/new_image_coordinates[2]), (new_image_coordinates[1]/new_image_coordinates[2])])

    error = np.sqrt((x[i] - new_image_coordinates_nonhomo[0])**2 + (y[i] - new_image_coordinates_nonhomo[1])**2)
    print("Reprojection error for point  =", error)

    reprojecton_error.append(error)
    
mean_error = np.mean(reprojecton_error)
print("\n")
print("Mean error = \n ", mean_error)
print("\n")




