
#include <cstdio>
#include <cstdlib>
#include <sdpa_call.h>
#include <chrono>

// The number of coefficients of our sdpa problem
#define DIMENSION 14

// The size of each matrix in the sdpa problem
#define BLOCK_SIZE 6

using std::cout;
using std::endl;

void printVector(const double *coefficients, int dimension);

void buildMat(SDPA &problem);

void runProblem(SDPA &sdpaProblem,
                double offsetFactor,
                double yMat[BLOCK_SIZE][BLOCK_SIZE],
                double xMat[BLOCK_SIZE][BLOCK_SIZE],
                double xVec[DIMENSION],
                bool copyToInitialPoint,
                std::vector<double> &coef);

void putCoefficients(SDPA &problem, const std::vector<double> &coef);

void setInitialPoint(SDPA &problem, int dim, int blockSize, double yMat[BLOCK_SIZE][BLOCK_SIZE], double xMat[BLOCK_SIZE][BLOCK_SIZE], double xVec[DIMENSION]);

int main() {
    // initialize general variables
    auto start_overall = chrono::steady_clock::now();
    long sum = 0;

    // The sdpa problem coefficients we are trying to solve
    std::vector<std::vector<double>> coefVectors =
            {
                    {0.0389107000, -0.03843660000, -0.3307035000, -0.2196378000, -0.1574861000, -0.1520822000, 0.08874700000, 0.3605860000, -0.2839050000, 0.1010212000, -0.5208514000, -0.0220351000, 0.1306196000, 0.2406502000}
            };

	// xMat + xVec + yMat are the the initial point inputBlockNumber
	// Fill them properly if you are planning to solve the sdpa problem with an initial point
    double xMat[BLOCK_SIZE][BLOCK_SIZE] = { {+1.3624559437946038e-01,-4.5204889778688922e-03,+7.5830404877687725e-03,+1.0796166049029132e-01,+2.6789637899233012e-03,+1.0581945139187109e-01 },
                                            {-4.5204889778688922e-03,+1.0796166049029132e-01,+2.6789637899233012e-03,-7.9012581990062648e-04,+4.2874455806629595e-03,-8.6261886761986376e-04 },
                                            {+7.5830404877687725e-03,+2.6789637899233012e-03,+1.0581945139187109e-01,+4.2874455806629595e-03,-8.6261886761986376e-04,+9.2649792391757101e-03 },
                                            {+1.0796166049029132e-01,-7.9012581990062648e-04,+4.2874455806629595e-03,+1.1137583709313288e-01,-1.7524849094959227e-03,+1.0317055199856009e-01 },
                                            {+2.6789637899233012e-03,+4.2874455806629595e-03,-8.6261886761986376e-04,-1.7524849094959227e-03,+1.0317055199856009e-01,+1.4981342544862425e-03 },
                                            {+1.0581945139187109e-01,-8.6261886761986376e-04,+9.2649792391757101e-03,+1.0317055199856009e-01,+1.4981342544862425e-03,+1.1847524076596173e-01 }   };
    double yMat[BLOCK_SIZE][BLOCK_SIZE] =    { {+1.9531881872946633e-01,+5.2603207200000021e-03,-4.3663586534999953e-03,-7.3658801723510497e-02,-3.9041372146494005e-03,-6.8087820381363989e-02 },
                                               {+5.2603207200000021e-03,+2.0715475870595390e-01,-5.8496426903506017e-03,-1.2937289950000015e-03,-9.5136145005070439e-03,-8.9859445547794633e-04 },
                                               {-4.3663586534999953e-03,-5.8496426903506017e-03,+2.1145436822166075e-01,+1.3288265405070573e-03,+2.0966801334779475e-03,-7.5748879599999968e-03 },
                                               {-7.3658801723510497e-02,-1.2937289950000015e-03,+1.3288265405070573e-03,+2.1350524052946640e-01,+4.2771443364999998e-03,-7.4898176339297823e-02 },
                                               {-3.9041372146494005e-03,-9.5136145005070439e-03,+2.0966801334779475e-03,+4.2771443364999998e-03,+2.1724104963752877e-01,-1.7539090065000014e-03 },
                                               {-6.8087820381363989e-02,-8.9859445547794633e-04,-7.5748879599999968e-03,-7.4898176339297823e-02,-1.7539090065000014e-03,+2.0590504792946640e-01 }   };
    double xVec[] = {-1.1847524076596173e-01,-1.4981342544862425e-03,-1.0317055199856009e-01,+1.7524849094959227e-03,-1.1137583709313288e-01,-9.2649792391757101e-03,+8.6261886761986376e-04,-4.2874455806629595e-03,+7.9012581990062648e-04,-1.0581945139187109e-01,-2.6789637899233012e-03,-1.0796166049029132e-01,-7.5830404877687725e-03,+4.5204889778688922e-03};

    int iterations =  10;
    int iteration;
	
	// Solving the same problem multiple times to get assurance
    for (iteration = 0; iteration < iterations; ++iteration) {
        // Init the problem
        SDPA sdpaProblem;
        sdpaProblem.setParameterType(SDPA::PARAMETER_DEFAULT);
        sdpaProblem.inputConstraintNumber(DIMENSION);
        sdpaProblem.inputBlockNumber(1);
        sdpaProblem.inputBlockSize(1, BLOCK_SIZE);
        sdpaProblem.inputBlockType(1, SDPA::SDP);
        sdpaProblem.initializeUpperTriangleSpace();
		
		// Switch it to true in order to use an initial point as a starting point
        bool initialPointEnabled = false;
        bool setInitialPointEveryNiterations = false;
        int n = 7;

        // set initial point if not first iteration
        if (initialPointEnabled && iteration > 0) {
            sdpaProblem.setInitPoint(true);
            setInitialPoint(sdpaProblem, DIMENSION, BLOCK_SIZE, yMat, xMat, xVec);
        }

        // choose whether the current SDPA problem should set its result as the initial point for the following problems
        bool copyToInitialPoint = false;
        if (setInitialPointEveryNiterations && initialPointEnabled && iteration % n == 0) {
            copyToInitialPoint = true;
        }

        std::cout << "Iteration number: " << iteration + 1 << std::endl;

        // run SPDA and measure time
        auto start = chrono::steady_clock::now();
        runProblem(sdpaProblem, 0, yMat, xMat, xVec, copyToInitialPoint, coefVectors[0]);
        auto end = chrono::steady_clock::now();
        sum += chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    }

    // print final statistics
    auto end_overall = chrono::steady_clock::now();
    std::cout << "average solving time (nanos): " << sum / iterations << std::endl;
    std::cout << "total time (seconds): " << chrono::duration_cast<chrono::seconds>(end_overall - start_overall).count()
              << std::endl;
    std::cout << "finished after " << iteration << " iterations" << std::endl;
    exit(0);
}

/*
 * sdpaProblem - problem to solve
 * offsetFactor - how much to increase the b's offset
 * yMat, xMat, xVec - output parameters for the next initial point
 * copyToInitialPoint - whether the problem should copy its results to yMat, xMat and xVec
 * coef - coefficients vector
 */
void runProblem(SDPA &sdpaProblem,
                double offsetFactor,
                double yMat[BLOCK_SIZE][BLOCK_SIZE],
                double xMat[BLOCK_SIZE][BLOCK_SIZE],
                double xVec[DIMENSION],
                bool copyToInitialPoint,
                std::vector<double> &coef) {

	// Inserting the coefficients to the problem
    putCoefficients(sdpaProblem, coef);
    buildMat(sdpaProblem);

    sdpaProblem.initializeUpperTriangle();
    sdpaProblem.initializeSolve();
    sdpaProblem.solve();

    // copy the sdpa result matrices in case an initial point is needed for next problems

    /*
    * NOTE - you can get the result coefficients by using sdpaProblem.getResultXVec() BEFORE you terminate the problem.
    */
    int matSize = BLOCK_SIZE * BLOCK_SIZE;
    if (copyToInitialPoint) {
        std::memcpy(yMat, sdpaProblem.getResultYMat(1), matSize * sizeof(double));
        std::memcpy(xMat, sdpaProblem.getResultXMat(1), matSize * sizeof(double));
        std::memcpy(xVec, sdpaProblem.getResultXVec(), DIMENSION * sizeof(double));
    }

    fprintf(stdout, "primal value: %3.10e\n", sdpaProblem.getPrimalObj());
    fprintf(stdout, "dual value: %3.10e\n", sdpaProblem.getDualObj());
    fprintf(stdout, "time (millis) : %f\n", sdpaProblem.getSolveTime() * 1000);
    fprintf(stdout, "iterations : %d\n", sdpaProblem.getIteration());

    sdpaProblem.terminate();
}


void setInitialPoint(SDPA &problem, int dim, int blockSize, double yMat[BLOCK_SIZE][BLOCK_SIZE], double xMat[BLOCK_SIZE][BLOCK_SIZE], double xVec[BLOCK_SIZE]) {
    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < blockSize; ++j) {
            problem.inputInitYMat(1, i + 1, j + 1, yMat[i][j]);
            problem.inputInitXMat(1, i + 1, j + 1, xMat[i][j]);
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

/*
 * Building the matrices for the corresponding coefficients
 */
void buildMat(SDPA &problem) {
    problem.inputElement(0,1,1,1,-1 );
    problem.inputElement(1,1,6,6,-1 );
    problem.inputElement(1,1,1,1,1  );
    problem.inputElement(2,1,6,5,-1 );
    problem.inputElement(3,1,6,4,-1 );
    problem.inputElement(3,1,5,5,-1 );
    problem.inputElement(3,1,1,1,2  );
    problem.inputElement(4,1,5,4,-1 );
    problem.inputElement(5,1,4,4,-1 );
    problem.inputElement(5,1,1,1,1  );
    problem.inputElement(6,1,6,3,-1 );
    problem.inputElement(7,1,6,2,-1 );
    problem.inputElement(7,1,5,3,-1 );
    problem.inputElement(8,1,5,2,-1 );
    problem.inputElement(8,1,4,3,-1 );
    problem.inputElement(9,1,4,2,-1 );
    problem.inputElement(10,1,6,1,-1);
    problem.inputElement(10,1,3,3,-1);
    problem.inputElement(10,1,1,1,2 );
    problem.inputElement(11,1,5,1,-1);
    problem.inputElement(11,1,3,2,-1);
    problem.inputElement(12,1,4,1,-1);
    problem.inputElement(12,1,2,2,-1);
    problem.inputElement(12,1,1,1,2 );
    problem.inputElement(13,1,3,1,-1);
    problem.inputElement(14,1,2,1,-1);
}

void printVector(const double *coefficients, int dimension) {
    for (int i = 0; i < dimension; ++i) {
        fprintf(stdout, "y%d = %.17g\n", i + 1, coefficients[i]);
    }
}
