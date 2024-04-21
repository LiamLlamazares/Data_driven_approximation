import numpy as np
from skfem import *
from skfem.helpers import dot, grad

# Define a 2D domain, for example, a unit square
mesh = MeshTri.init_symmetric()

# Use an elementary finite element; piecewise linear basis functions
element = ElementTriP1()

# Create a basis corresponding to this element and mesh
basis = InteriorBasis(mesh, element)


def point_in_triangle(p, triangle):
    """Check if point p is inside the triangle with given vertices."""

    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] -
                                                                      p3[1])

    b1 = sign(p, triangle[0], triangle[1]) < 0.0
    b2 = sign(p, triangle[1], triangle[2]) < 0.0
    b3 = sign(p, triangle[2], triangle[0]) < 0.0

    return ((b1 == b2) and (b2 == b3))


# Sampling m points from the domain
m = 100
points = np.random.rand(m, 2)  # Random points in [0, 1]^2

# Initialize the matrices
C = np.zeros((basis.N, basis.N))
G = np.zeros((basis.N, basis.N))

# Manual element finding
for pt in points:
    found = False
    for i in range(mesh.t.shape[1]):
        vertices = mesh.p[:, mesh.t[:, i]].T
        if point_in_triangle(pt, vertices):
            elem = i
            found = True
            break
    if not found:
        continue
mesh.t2f
    # Get local coordinates in the reference element
    local = basis.mapping.invF(elem, pt.reshape(1, -1))
    phi = basis.elem_basis(elem)(local).flatten()
    grad_phi = basis.elem_grads(elem)(local).reshape(-1, 2)

    # Build the matrices
    C += np.outer(phi, phi)
    G += (grad_phi @ grad_phi.T)  # Correcting matrix multiplication

# Print or use the matrices C and G
print("Approximated Mass Matrix (C):", C)
print("Approximated Stiffness Matrix (G):", G)
