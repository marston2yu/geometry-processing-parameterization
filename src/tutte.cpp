#include "tutte.h"
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>
#include <igl/edges.h>
#include <vector>
void tutte(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    Eigen::MatrixXd &U)
{
  // Replace with your code
  int vertexNumber = V.rows();
  // Map boundary vertices to UV plane.
  Eigen::VectorXi boundaryIndices;
  igl::boundary_loop(F, boundaryIndices);
  Eigen::MatrixXd boundaryCircle;
  boundaryCircle = boundaryCircle;
  igl::map_vertices_to_circle(V, boundaryIndices, boundaryCircle);
  // Minimize quadratic energy function subject to boundary condition.
  // Problem: tr(1/2 * U'*A*U + U'*B) where B is zero vector.
  //          subject to U(boundaryIndices, :) = boundaryCircle, Aeq * U = Beq
  //          where Aeq is zero matrix and Beq is zero vector.

  // Calculate discrete laplacian matrix as weight matrix.
  Eigen::SparseMatrix<double> A(vertexNumber, vertexNumber);
  igl::cotmatrix(V, F, A);

  // Another choice is using distance as weight matrix.
  // Loop over edges to calculate weights for edges.
  Eigen::MatrixXi edges;
  igl::edges(F, edges);

  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(edges.rows() * 4);
  Eigen::SparseMatrix<double> L;
  for (int i = 0; i < edges.rows(); i++)
  {
    int pi = edges(i, 0);
    int pj = edges(i, 1);
    double wij = 1.0 / (V.row(pi) - V.row(pj)).norm();

    tripletList.push_back(T(pi, pj, wij));
    tripletList.push_back(T(pj, pi, wij));

    // Duplicates will be summed up.
    tripletList.push_back(T(pi, pi, -wij));
    tripletList.push_back(T(pj, pj, -wij));
  }
  L.resize(vertexNumber, vertexNumber);
  L.setFromTriplets(tripletList.begin(), tripletList.end());

  // Empty constraints just to satisfy function api.
  Eigen::VectorXd B(vertexNumber);
  Eigen::SparseMatrix<double> Aeq;
  Eigen::VectorXd Beq;
  
  // Calculate mapping with minimizing energy fuction.
  // Weight matrix can be L or A.
  igl::min_quad_with_fixed(L, B, boundaryIndices, boundaryCircle, Aeq, Beq, false, U);
}
