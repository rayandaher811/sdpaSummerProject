
#include <cstdio>
#include <cstdlib>
#include <sdpa_call.h>
#include <chrono>

#define DIMENSION 35
#define BLOCK_SIZE 11

using std::cout;
using std::endl;

void printVector(const double *coefficients, int dimension);

void buildMat(SDPA &problem);

void runProblem(SDPA &problem, double offset, double *yMat, double *xMat, double *xVec, bool copyToInitialPoint);

void putCoefficients(SDPA &problem, const std::vector<double> &coef);

void printStatistics(SDPA &problem, int mDim);

void setInitialPoint(SDPA &problem, int dim, int blockSize, double *yMat, double *xMat, double *xVec);

int main() {
    // initialize general variables
    auto start_overall = chrono::steady_clock::now();
    long sum = 0;

    // current initial point
    double *yMat = new double[11 * 11];
    double *xMat = new double[11 * 11];
    double *xVec = new double[35];

    int iterations = 20;
    int iteration;
    for (iteration = 0; iteration < iterations; ++iteration) {
        // Init problem
        SDPA sdpaProblem;
        sdpaProblem.setParameterType(SDPA::PARAMETER_DEFAULT);

        sdpaProblem.inputConstraintNumber(DIMENSION);
        sdpaProblem.inputBlockNumber(1);
        sdpaProblem.inputBlockSize(1, BLOCK_SIZE);
        sdpaProblem.inputBlockType(1, SDPA::SDP);
        sdpaProblem.initializeUpperTriangleSpace();

        // set initial point if not first iteration
        if (iteration > 0) {
           // setInitialPoint(sdpaProblem, DIMENSION, BLOCK_SIZE, yMat, xMat, xVec);
        }

        // choose whether the current SDPA should copy its results to the current initial point
        bool copyToInitialPoint = false;
        if (iteration == 0) {
            copyToInitialPoint = true;
        }

        // run SPDA and measure time
        auto start = chrono::steady_clock::now();
        runProblem(sdpaProblem, 0, yMat, xMat, xVec, copyToInitialPoint);
        auto end = chrono::steady_clock::now();
        sum += chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    }

    auto end_overall = chrono::steady_clock::now();
    std::cout << "average solving time (nanos): " << sum / iterations << std::endl;
    std::cout << "total time (seconds): " << chrono::duration_cast<chrono::seconds>(end_overall - start_overall).count()
              << std::endl;
    std::cout << "finished after " << iteration << " iterations" << std::endl;
    exit(0);
};

/*
 * sdpaProblem - problem to solve
 * offsetFactor - how much to increase the b's offset
 * yMat, xMat, xVec - output parameters for the next initial point
 * copyToInitialPoint - whether the problem should copy its results to yMat, xMat and xVec
 */
void runProblem(SDPA &sdpaProblem,
                double offsetFactor,
                double *yMat,
                double *xMat,
                double *xVec,
                bool copyToInitialPoint) {

    /* uncomment to display info on each SDPA run */
    //sdpaProblem.setDisplay(stdout);

    putCoefficients(sdpaProblem, {
            -6.719531000, -2.431764000, -5.839611000, 1.002536000, -5.030302000, -2.198534000, 2.811224000,
            -5.877316000, -0.4867120000, -3.418229000, 5.990764000, -2.916759000, -4.123538000, 0.8149200000,
            -3.985329000, -1.657860000, 1.708110000, -5.287228000, 1.573062000, -0.8551160000, -3.270076000,
            -0.9646860000, 5.284532000, -0.8993340000, 0.7142700000, -5.961875000, 1.220422000, -3.968850000,
            -3.013936000, 1.312744000, -6.833562000, 3.614576000, -1.889932000, -2.339266000, -7.698171000
    });
    buildMat(sdpaProblem);

    sdpaProblem.initializeUpperTriangle();
    sdpaProblem.initializeSolve();
    sdpaProblem.solve();

    int matSize = BLOCK_SIZE * BLOCK_SIZE;
    if (copyToInitialPoint) {
        std::memcpy(yMat, sdpaProblem.getResultYMat(1), matSize * sizeof(double));
        std::memcpy(xMat, sdpaProblem.getResultXMat(1), matSize * sizeof(double));
        std::memcpy(xVec, sdpaProblem.getResultXVec(), DIMENSION * sizeof(double));
    }

    fprintf(stdout, "primal value: %e\n", sdpaProblem.getPrimalObj());
    fprintf(stdout, "dual value: %e\n", sdpaProblem.getDualObj());

    sdpaProblem.terminate();
}


void setInitialPoint(SDPA &problem, int dim, int blockSize, double *yMat, double *xMat, double *xVec) {
    for (int i = 0; i < blockSize; ++i) {
        for (int j = i; j < blockSize; ++j) {
            if (yMat[i + blockSize * j] != 0) {
                // std::cout << "(" << i << ", " << j << ") = " << yMat[i + blockSize * j] << std::endl;
                problem.inputInitYMat(1, i + 1, j + 1, yMat[i + blockSize * j]);
            }
        }
    }

    for (int i = 0; i < blockSize; ++i) {
        for (int j = i; j < blockSize; ++j) {
            if (xMat[i + blockSize * j] != 0) {
                // std::cout << "(" << i << ", " << j << ") = " << xMat[i + blockSize * j] << std::endl;
                problem.inputInitXMat(1, i + 1, j + 1, xMat[i + blockSize * j]);
            }
        }
    }

    for (int i = 0; i < dim; ++i) {
        problem.inputInitXVec(i + 1, xVec[i]);
    }
}

void putCoefficients(SDPA &problem, const std::vector<double> &coef) {
    for (int i = 1; i <= coef.size(); ++i) {
        problem.inputCVec(i, coef.at(i - 1));
    }
}

void buildMat(SDPA &problem) {
    problem.inputElement(0, 1, 1, 1, 1);
    problem.inputElement(1, 1, 1, 1, -1);
    problem.inputElement(1, 1, 11, 11, -1);
    problem.inputElement(2, 1, 11, 10, -1);
    problem.inputElement(3, 1, 1, 1, -2);
    problem.inputElement(3, 1, 10, 10, -1);
    problem.inputElement(3, 1, 11, 9, -1);
    problem.inputElement(4, 1, 10, 9, -1);
    problem.inputElement(5, 1, 1, 1, -1);
    problem.inputElement(5, 1, 9, 9, -1);
    problem.inputElement(6, 1, 11, 8, -1);
    problem.inputElement(7, 1, 10, 8, -1);
    problem.inputElement(7, 1, 11, 7, -1);
    problem.inputElement(8, 1, 9, 8, -1);
    problem.inputElement(8, 1, 10, 7, -1);
    problem.inputElement(9, 1, 9, 7, -1);
    problem.inputElement(10, 1, 1, 1, -2);
    problem.inputElement(10, 1, 8, 8, -1);
    problem.inputElement(10, 1, 11, 6, -1);
    problem.inputElement(11, 1, 8, 7, -1);
    problem.inputElement(11, 1, 10, 6, -1);
    problem.inputElement(12, 1, 1, 1, -2);
    problem.inputElement(12, 1, 7, 7, -1);
    problem.inputElement(12, 1, 9, 6, -1);
    problem.inputElement(13, 1, 8, 6, -1);
    problem.inputElement(14, 1, 7, 6, -1);
    problem.inputElement(15, 1, 1, 1, -1);
    problem.inputElement(15, 1, 6, 6, -1);
    problem.inputElement(16, 1, 11, 5, -1);
    problem.inputElement(17, 1, 10, 5, -1);
    problem.inputElement(17, 1, 11, 4, -1);
    problem.inputElement(18, 1, 9, 5, -1);
    problem.inputElement(18, 1, 10, 4, -1);
    problem.inputElement(19, 1, 9, 4, -1);
    problem.inputElement(20, 1, 8, 5, -1);
    problem.inputElement(20, 1, 11, 3, -1);
    problem.inputElement(21, 1, 7, 5, -1);
    problem.inputElement(21, 1, 8, 4, -1);
    problem.inputElement(21, 1, 10, 3, -1);
    problem.inputElement(22, 1, 7, 4, -1);
    problem.inputElement(22, 1, 9, 3, -1);
    problem.inputElement(23, 1, 6, 5, -1);
    problem.inputElement(23, 1, 8, 3, -1);
    problem.inputElement(24, 1, 6, 4, -1);
    problem.inputElement(24, 1, 7, 3, -1);
    problem.inputElement(25, 1, 6, 3, -1);
    problem.inputElement(26, 1, 1, 1, -2);
    problem.inputElement(26, 1, 5, 5, -1);
    problem.inputElement(26, 1, 11, 2, -1);
    problem.inputElement(27, 1, 5, 4, -1);
    problem.inputElement(27, 1, 10, 2, -1);
    problem.inputElement(28, 1, 1, 1, -2);
    problem.inputElement(28, 1, 4, 4, -1);
    problem.inputElement(28, 1, 9, 2, -1);
    problem.inputElement(29, 1, 5, 3, -1);
    problem.inputElement(29, 1, 8, 2, -1);
    problem.inputElement(30, 1, 4, 3, -1);
    problem.inputElement(30, 1, 7, 2, -1);
    problem.inputElement(31, 1, 1, 1, -2);
    problem.inputElement(31, 1, 3, 3, -1);
    problem.inputElement(31, 1, 6, 2, -1);
    problem.inputElement(32, 1, 5, 2, -1);
    problem.inputElement(33, 1, 4, 2, -1);
    problem.inputElement(34, 1, 3, 2, -1);
    problem.inputElement(35, 1, 1, 1, -1);
    problem.inputElement(35, 1, 2, 2, -1);
}

void printVector(const double *coefficients, int dimension) {
    for (int i = 0; i < dimension; ++i) {
        fprintf(stdout, "y%d = %.17g\n", i + 1, coefficients[i]);
    }
}

void printStatistics(SDPA &problem, int mDim) {
    fprintf(stdout, "time  %.17g\n", problem.getSolveTime());
    fprintf(stdout, "primsal value %.17g\n", problem.getPrimalObj());
    fprintf(stdout, "dual value %.17g\n", problem.getDualObj());
    printVector(problem.getResultXVec(), mDim);
}
