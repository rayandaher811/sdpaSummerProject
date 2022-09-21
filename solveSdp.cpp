
#include <cstdio>
#include <cstdlib>
#include <sdpa_call.h>
#include <chrono>

#define DIMENSION 14
#define BLOCK_SIZE 6

using std::cout;
using std::endl;

void printVector(const double *coefficients, int dimension);

void buildMat(SDPA &problem);

void runProblem(SDPA &problem, double offset, double *yMat, double *xMat, double *xVec, bool copyToInitialPoint,
                std::vector<double> &coef);

void putCoefficients(SDPA &problem, const std::vector<double> &coef);

void setInitialPoint(SDPA &problem, int dim, int blockSize, double *yMat, double *xMat, double *xVec);

std::vector<double> extractSmallProblem(const std::vector<double> &coef, double b);

int main() {
    // initialize general variables
    auto start_overall = chrono::steady_clock::now();
    long sum = 0;
    long sumCoefPrep = 0;

    // initial inputs
    std::vector<std::vector<double>> coefVectors =
            {
                {
                        0.7739754000, -0.4607657000, 0.7262255000, 0.2417991000, 0.8217785000, -0.2861407000, 0.1378471000, 0.1873807000, 0.2716400000, 0.9810004000, -0.5863106000, 1.020847450, 0.2138725000, 0.06236740000, 0.9385016500, 0.04550430000, -0.4071891000, 0.1365847000, 0.05756750000, -0.5618737000, -0.2161587000, 0.1953773000, -0.5688474000, -0.04414210000, 0.3536658000, 0.8546904500, -0.3303616000, 0.9372889000, 0.1940359000, 0.4774460000, 1.467044000, -0.04947000000, 0.2098904000, -0.1627388000, 0.8149880500
                }
            };

    // current initial point
    double *yMat = new double[BLOCK_SIZE * BLOCK_SIZE];
    double *xMat = new double[BLOCK_SIZE * BLOCK_SIZE];
    double *xVec = new double[DIMENSION];

    int iterations =  1;
    int iteration;
    for (iteration = 0; iteration < iterations; ++iteration) {

        std::cout << "Iteration number: " << iteration + 1 << std::endl;

        double minPrimal = 300000000;
        double maxPrimal = -300000000;
        double minDual = 300000000;
        double maxDual = -300000000;

        for (double b = -100; b <= 100; b+=0.01) {
            // Init problem
            SDPA sdpaProblem;
            sdpaProblem.setParameterType(SDPA::PARAMETER_DEFAULT);
            sdpaProblem.inputConstraintNumber(DIMENSION);
            sdpaProblem.inputBlockNumber(1);
            sdpaProblem.inputBlockSize(1, BLOCK_SIZE);
            sdpaProblem.inputBlockType(1, SDPA::SDP);
            sdpaProblem.initializeUpperTriangleSpace();
            bool initialPointEnabled = false;

            // set initial point if not first iteration
            if (initialPointEnabled && iteration > 0) {
                sdpaProblem.setInitPoint(true);
                setInitialPoint(sdpaProblem, DIMENSION, BLOCK_SIZE, yMat, xMat, xVec);
            }

            // choose whether the current SDPA problem should set its result as the initial point for the following problems
            bool copyToInitialPoint = false;
            if (initialPointEnabled && iteration % 3 == 0) {
                copyToInitialPoint = true;
            }


            // run SPDA and measure time

            auto startCoefPrep = chrono::steady_clock::now();
            std:: vector<double> smallerProblem = extractSmallProblem(coefVectors[0], b);
            auto endCoefPrep = chrono::steady_clock::now();

            auto start = chrono::steady_clock::now();
            runProblem(sdpaProblem, 0, yMat, xMat, xVec, copyToInitialPoint, smallerProblem);
            auto end = chrono::steady_clock::now();

            sum += chrono::duration_cast<chrono::nanoseconds>(end - start).count();
            sumCoefPrep += chrono::duration_cast<chrono::nanoseconds>(endCoefPrep - startCoefPrep).count();

            if(minPrimal > sdpaProblem.getPrimalObj())
                minPrimal = sdpaProblem.getPrimalObj();
            if(maxPrimal < sdpaProblem.getPrimalObj())
                maxPrimal = sdpaProblem.getPrimalObj();
            if(minDual > sdpaProblem.getDualObj())
                minDual = sdpaProblem.getDualObj();
            if(maxDual < sdpaProblem.getDualObj())
                maxDual = sdpaProblem.getDualObj();
        }

        std::cout << "Real minimum: " << minPrimal + coefVectors[0][34] << std::endl;
        std::cout << "min primal: " << minPrimal << std::endl;
        std::cout << "max primal: " << maxPrimal << std::endl;
        std::cout << "min dual: " << minDual << std::endl;
        std::cout << "max dual: " << maxDual << std::endl;
    }

    // print final statistics
    auto end_overall = chrono::steady_clock::now();
    std::cout << "average sdpa solving time (nanos): " << sum / iterations << std::endl;
    std::cout << "average coef preparation time (nanos): " << sumCoefPrep / iterations << std::endl;
    std::cout << "average problem solving time (nanos): " << (sum + sumCoefPrep) / iterations << std::endl;
    std::cout << "Total problems solving time (nanos): " << sum<< std::endl;
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
                bool copyToInitialPoint,
                std::vector<double> &coef) {

    /* uncomment to display info on each SDPA run */
    //sdpaProblem.setDisplay(stdout);

    putCoefficients(sdpaProblem, coef);
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

    fprintf(stdout, "primal value: %3.10e\n", sdpaProblem.getPrimalObj());
    fprintf(stdout, "dual value: %3.10e\n", sdpaProblem.getDualObj());
    fprintf(stdout, "time (millis) : %f\n", sdpaProblem.getSolveTime() * 1000);
    fprintf(stdout, "iterations : %d\n", sdpaProblem.getIteration());

    sdpaProblem.terminate();
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
    problem.inputElement(0,1,1,1,-1);
    problem.inputElement(1,1,6,6,-1);
    problem.inputElement(1,1,1,1,1);
    problem.inputElement(2,1,6,5,-1);
    problem.inputElement(3,1,6,4,-1);
    problem.inputElement(3,1,5,5,-1);
    problem.inputElement(3,1,1,1,2);
    problem.inputElement(4,1,5,4,-1);
    problem.inputElement(5,1,4,4,-1);
    problem.inputElement(5,1,1,1,1);
    problem.inputElement(6,1,6,3,-1);
    problem.inputElement(7,1,6,2,-1);
    problem.inputElement(7,1,5,3,-1);
    problem.inputElement(8,1,5,2,-1);
    problem.inputElement(8,1,4,3,-1);
    problem.inputElement(9,1,4,2,-1);
    problem.inputElement(10,1,6,1,-1);
    problem.inputElement(10,1,3,3,-1);
    problem.inputElement(10,1,1,1,2);
    problem.inputElement(11,1,5,1,-1);
    problem.inputElement(11,1,3,2,-1);
    problem.inputElement(12,1,4,1,-1);
    problem.inputElement(12,1,2,2,-1);
    problem.inputElement(12,1,1,1,2);
    problem.inputElement(13,1,3,1,-1);
    problem.inputElement(14,1,2,1,-1);
}

void printVector(const double *coefficients, int dimension) {
    for (int i = 0; i < dimension; ++i) {
        fprintf(stdout, "y%d = %.17g\n", i + 1, coefficients[i]);
    }
}

std::vector<double> extractSmallProblem(const std::vector<double> &coef, double b){
    double b400 = coef[34];
    double b310 = coef[33];
    double b301 = coef[32]/sqrt(b*b+1.0)+coef[31]*b/sqrt(b*b+1.0);
    double b220 = coef[30];
    double b211 = coef[29]/sqrt(b*b+1.0)+coef[28]*b/sqrt(b*b+1.0);
    double b202 = coef[27]/(b*b+1.0)+coef[26]/(b*b+1.0)*b+coef[25]*b*b/(b*b+1.0);
    double b130 = coef[24];
    double b121 = coef[23]/sqrt(b*b+1.0)+coef[22]*b/sqrt(b*b+1.0);
    double b112 = coef[21]/(b*b+1.0)+coef[19]*b*b/(b*b+1.0)+coef[20]/(b*b+1.0)*b;
    double b103 = coef[18]/sqrt(pow(b*b+1.0,3.0))+coef[17]/sqrt(pow(b*b+1.0,3.0))*b+coef[16]/sqrt(pow(b*b+1.0,3.0))*b*b+coef[15]*b*b*b/sqrt(pow(b*b+1.0,3.0));
    double b040 = coef[14];
    double b031 = coef[13]/sqrt(b*b+1.0)+coef[12]*b/sqrt(b*b+1.0);
    double b022 = coef[11]/(b*b+1.0)+coef[10]/(b*b+1.0)*b+coef[9]*b*b/(b*b+1.0);
    double b013 = coef[8]/sqrt(pow(b*b+1.0,3.0))+coef[6]/sqrt(pow(b*b+1.0,3.0))*b*b+coef[7]/sqrt(pow(b*b+1.0,3.0))*b+coef[5]*b*b*b/sqrt(pow(b*b+1.0,3.0));
    double b004 = coef[1]/pow(b*b+1.0,2.0)*b*b*b+coef[2]/pow(b*b+1.0,2.0)*b*b+coef[3]/pow(b*b+1.0,2.0)*b+coef[0]*b*b*b*b/pow(b*b+1.0,2.0)+coef[4]/pow(b*b+1.0,2.0);

    return {-b004 + b400,
            -b013,
            -b022 + 2*b400,
            -b031,
            -b040 + b400,
            -b103,
            -b112,
            -b121,
            -b130,
            -b202 + 2*b400,
            -b211,
            -b220 + 2*b400,
            -b301,
            -b310
    };
}
