
#include <cstdio>
#include <cstdlib>
#include <sdpa_call.h>
#include <chrono>
#include <future>

#define DIMENSION 14
#define BLOCK_SIZE 6

using std::cout;
using std::endl;

void printVector(const double *coefficients, int dimension);

void buildMat(SDPA &problem);

void runProblem(SDPA &problem, double offset, double *yMat, double *xMat, double *xVec,
                std::vector<double> &coef);

double runProblemsRange(double bStart, double bEnd, int iteration);

void putCoefficients(SDPA &problem, const std::vector<double> &coef);

void setInitialPoint(SDPA &problem, int dim, int blockSize, double *yMat, double *xMat, double *xVec);

std::vector<double> extractSmallProblem(const std::vector<double> &coef, double b);

// Matrices configs
double *yMat = new double[BLOCK_SIZE * BLOCK_SIZE];
double *xMat = new double[BLOCK_SIZE * BLOCK_SIZE];
double *xVec = new double[DIMENSION];

// b configs
double bSTART = -2;
double bEND = 2;
double bJUMPS = 0.01;

// Threads count
int threadAmount =4;

// Time measurements sums
long sum = 0;
long sumCoefPrep = 0;


// initial inputs
std::vector<std::vector<double>> coefVectors =
        {
                {
                        0.8267088500, -0.03762930000, 1.148176250, 0.3280084000, 0.7279020500, -0.2969960000,
                        -0.4217697000, -0.2741615000, -0.01951170000, 0.4880662500, -0.3687674000, 0.7530890500,
                        0.3540255000, -0.2344946000, 0.7331724500, 0.1567501000, -0.07656680000, -0.08375230000,
                        -0.5491424000, -0.5935686000, 0.3219790000, 0.4282622000, -0.3411265000, -0.1592065000,
                        0.1864248000, 0.9071691500, -0.04976520000, 0.7128398500, 1.009891000, -0.2623729000,
                        0.8567707500, 0.04297800000, -0.06063250000, -0.09147550000, 0.7973897000
                }
        };

int main() {
    // initialize general variables
    auto start_overall = chrono::steady_clock::now();
    auto startProblemSolving = chrono::steady_clock::now();
    auto endProblemSolving = chrono::steady_clock::now();

    int iterations = 1;
    int iteration;
    double bThreadJumps = (bEND - bSTART) / (threadAmount);

    for (iteration = 0; iteration < iterations; ++iteration) {

        std::cout << "Iteration number: " << iteration + 1 << std::endl;
        startProblemSolving = chrono::steady_clock::now();
        double minPrimal = 3000000000;
        std::vector<std::future<double>> allThreads;
        int threadsCreated =0;

        for (double start = bSTART; start < bEND; start += bThreadJumps) {
            threadsCreated++;
            allThreads.push_back(std::async(runProblemsRange, start, start + bThreadJumps, iteration));
        }
        std::cout << "Threads created: " << threadsCreated << std::endl;

        std::cout << "Threads done: " << std::endl;
        for(auto &thread: allThreads) {
            if(thread.get() < minPrimal)
                minPrimal = thread.get();
        }

        endProblemSolving = chrono::steady_clock::now();

        std::cout << "Real minimum: " << minPrimal + coefVectors[iteration][34] << std::endl;
        std::cout << "Minimum primal: " << minPrimal << std::endl;
    }

    // print final statistics
    auto end_overall = chrono::steady_clock::now();
    auto problemsSolvingDuration = chrono::duration_cast<chrono::nanoseconds>(
            endProblemSolving - startProblemSolving).count();
    auto smallProblemsSolvedAmount = (bEND - bSTART) * iterations / bJUMPS;

    std::cout << "average sdpa solving time (nanos): " << sum / smallProblemsSolvedAmount << std::endl;
    std::cout << "average coef preparation time (nanos): " << sumCoefPrep / smallProblemsSolvedAmount << std::endl;
    std::cout << "average small problem solving time (nanos): " << (sum + sumCoefPrep) / smallProblemsSolvedAmount
              << std::endl;
    std::cout << "average big problem solving time (nanos): " << problemsSolvingDuration / iterations << std::endl;
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
                double *yMat,
                double *xMat,
                double *xVec,
                 std::vector<double> &coef) {
    /* uncomment to display info on each SDPA run */
    //sdpaProblem.setDisplay(stdout);

    putCoefficients(sdpaProblem, coef);

    sdpaProblem.initializeUpperTriangle();
    sdpaProblem.initializeSolve();
    sdpaProblem.solve();

//    fprintf(stdout, "primal value: %3.10e\n", sdpaProblem.getPrimalObj());
//    fprintf(stdout, "dual value: %3.10e\n", sdpaProblem.getDualObj());
//    fprintf(stdout, "time (millis) : %f\n", sdpaProblem.getSolveTime() * 1000);


    sdpaProblem.terminate();
}

double runProblemsRange(double bStart, double bEnd, int iteration) {

    double minPrimal = 300000000;
    double maxPrimal = -300000000;
    double minDual = 300000000;
    double maxDual = -300000000;

    std::future<double> f1;
    for (double b = bStart; b <= bEnd; b += bJUMPS) {

        // run SPDA and measure time
        auto startCoefPrep = chrono::steady_clock::now();

        // Init problem
        SDPA sdpaProblem;
        sdpaProblem.setParameterType(SDPA::PARAMETER_DEFAULT);
        sdpaProblem.inputConstraintNumber(DIMENSION);
        sdpaProblem.inputBlockNumber(1);
        sdpaProblem.inputBlockSize(1, BLOCK_SIZE);
        sdpaProblem.inputBlockType(1, SDPA::SDP);
        sdpaProblem.initializeUpperTriangleSpace();
        buildMat(sdpaProblem);

        std::cout << "Current b value: " << b << std::endl;

        std::vector<double> smallerProblem = extractSmallProblem(coefVectors[iteration], b);
        auto endCoefPrep = chrono::steady_clock::now();

        auto start = chrono::steady_clock::now();
        runProblem(sdpaProblem, 0, yMat, xMat, xVec, smallerProblem);
        auto end = chrono::steady_clock::now();

        sum += chrono::duration_cast<chrono::nanoseconds>(end - start).count();
        sumCoefPrep += chrono::duration_cast<chrono::nanoseconds>(endCoefPrep - startCoefPrep).count();

        if (minPrimal > sdpaProblem.getPrimalObj())
            minPrimal = sdpaProblem.getPrimalObj();
        if (maxPrimal < sdpaProblem.getPrimalObj())
            maxPrimal = sdpaProblem.getPrimalObj();
        if (minDual > sdpaProblem.getDualObj())
            minDual = sdpaProblem.getDualObj();
        if (maxDual < sdpaProblem.getDualObj())
            maxDual = sdpaProblem.getDualObj();

    }

    return minPrimal;
}

void setInitialPoint(SDPA &problem, int dim, int blockSize, double *yMat, double *xMat, double *xVec) {
    for (int i = 0; i < blockSize; ++i) {
        for (int j = i; j < blockSize; ++j) {
            if (yMat[i + blockSize * j] != 0) {
                //std::cout << "(" << i << ", " << j << ") = " << yMat[i + blockSize * j] << std::endl;
                problem.inputInitYMat(1, i + 1, j + 1, yMat[i + blockSize * j]);
            }
        }
    }

    for (int i = 0; i < blockSize; ++i) {
        for (int j = i; j < blockSize; ++j) {
            if (xMat[i + blockSize * j] != 0) {
                //std::cout << "(" << i << ", " << j << ") = " << xMat[i + blockSize * j] << std::endl;
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
    problem.inputElement(0, 1, 1, 1, -1);
    problem.inputElement(1, 1, 6, 6, -1);
    problem.inputElement(1, 1, 1, 1, 1);
    problem.inputElement(2, 1, 6, 5, -1);
    problem.inputElement(3, 1, 6, 4, -1);
    problem.inputElement(3, 1, 5, 5, -1);
    problem.inputElement(3, 1, 1, 1, 2);
    problem.inputElement(4, 1, 5, 4, -1);
    problem.inputElement(5, 1, 4, 4, -1);
    problem.inputElement(5, 1, 1, 1, 1);
    problem.inputElement(6, 1, 6, 3, -1);
    problem.inputElement(7, 1, 6, 2, -1);
    problem.inputElement(7, 1, 5, 3, -1);
    problem.inputElement(8, 1, 5, 2, -1);
    problem.inputElement(8, 1, 4, 3, -1);
    problem.inputElement(9, 1, 4, 2, -1);
    problem.inputElement(10, 1, 6, 1, -1);
    problem.inputElement(10, 1, 3, 3, -1);
    problem.inputElement(10, 1, 1, 1, 2);
    problem.inputElement(11, 1, 5, 1, -1);
    problem.inputElement(11, 1, 3, 2, -1);
    problem.inputElement(12, 1, 4, 1, -1);
    problem.inputElement(12, 1, 2, 2, -1);
    problem.inputElement(12, 1, 1, 1, 2);
    problem.inputElement(13, 1, 3, 1, -1);
    problem.inputElement(14, 1, 2, 1, -1);
}

void printVector(const double *coefficients, int dimension) {
    for (int i = 0; i < dimension; ++i) {
        fprintf(stdout, "y%d = %.17g\n", i + 1, coefficients[i]);
    }
}

std::vector<double> extractSmallProblem(const std::vector<double> &coef, double b) {
    double b400 = coef[34];
    double b310 = coef[33];
    double b301 = coef[32] / sqrt(b * b + 1.0) + coef[31] * b / sqrt(b * b + 1.0);
    double b220 = coef[30];
    double b211 = coef[29] / sqrt(b * b + 1.0) + coef[28] * b / sqrt(b * b + 1.0);
    double b202 = coef[27] / (b * b + 1.0) + coef[26] / (b * b + 1.0) * b + coef[25] * b * b / (b * b + 1.0);
    double b130 = coef[24];
    double b121 = coef[23] / sqrt(b * b + 1.0) + coef[22] * b / sqrt(b * b + 1.0);
    double b112 = coef[21] / (b * b + 1.0) + coef[19] * b * b / (b * b + 1.0) + coef[20] / (b * b + 1.0) * b;
    double b103 = coef[18] / sqrt(pow(b * b + 1.0, 3.0)) + coef[17] / sqrt(pow(b * b + 1.0, 3.0)) * b +
                  coef[16] / sqrt(pow(b * b + 1.0, 3.0)) * b * b + coef[15] * b * b * b / sqrt(pow(b * b + 1.0, 3.0));
    double b040 = coef[14];
    double b031 = coef[13] / sqrt(b * b + 1.0) + coef[12] * b / sqrt(b * b + 1.0);
    double b022 = coef[11] / (b * b + 1.0) + coef[10] / (b * b + 1.0) * b + coef[9] * b * b / (b * b + 1.0);
    double b013 = coef[8] / sqrt(pow(b * b + 1.0, 3.0)) + coef[6] / sqrt(pow(b * b + 1.0, 3.0)) * b * b +
                  coef[7] / sqrt(pow(b * b + 1.0, 3.0)) * b + coef[5] * b * b * b / sqrt(pow(b * b + 1.0, 3.0));
    double b004 = coef[1] / pow(b * b + 1.0, 2.0) * b * b * b + coef[2] / pow(b * b + 1.0, 2.0) * b * b +
                  coef[3] / pow(b * b + 1.0, 2.0) * b + coef[0] * b * b * b * b / pow(b * b + 1.0, 2.0) +
                  coef[4] / pow(b * b + 1.0, 2.0);

    return {-b004 + b400,
            -b013,
            -b022 + 2 * b400,
            -b031,
            -b040 + b400,
            -b103,
            -b112,
            -b121,
            -b130,
            -b202 + 2 * b400,
            -b211,
            -b220 + 2 * b400,
            -b301,
            -b310
    };
}
