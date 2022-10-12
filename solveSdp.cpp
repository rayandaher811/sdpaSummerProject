
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

void runProblem(SDPA &problem, double offset, double yMat[BLOCK_SIZE][BLOCK_SIZE], double xMat[BLOCK_SIZE][BLOCK_SIZE], double xVec[DIMENSION],
                std::vector<double> &coef);

void putCoefficients(SDPA &problem, const std::vector<double> &coef);

void setInitialPoint(SDPA &problem, int dim, int blockSize, double yMat[BLOCK_SIZE][BLOCK_SIZE], double xMat[BLOCK_SIZE][BLOCK_SIZE], double xVec[DIMENSION]);
void setInitialPoint(SDPA &problem, int dim, int blockSize, double *yMat, double *xMat, double *xVec);
void printMatrices(SDPA &sdpaProblem);


std::vector<double> extractSmallProblem(const std::vector<double> &coef, double b);
int main(int argc, char **argv) {
    // initialize general variables
    auto start_overall = chrono::steady_clock::now();
    long sum = 0;
    long sumCoefPrep = 0;
    auto startProblemSolving = chrono::steady_clock::now();
    auto endProblemSolving = chrono::steady_clock::now();

    double *bZeroXMat =  new double [BLOCK_SIZE*BLOCK_SIZE]{0.09436, -0.0538021, 0.125528, 0.0379947, -0.0727448, 0.173503, -0.0538021, 0.0379947, -0.0727448, -0.0221038, 0.0488784, -0.101856, 0.125528, -0.0727448, 0.173503, 0.0488784, -0.101856, 0.235204, 0.0379947, -0.0221038, 0.0488784, 0.0185714, -0.0274035, 0.069665, -0.0727448, 0.0488784, -0.101856, -0.0274035, 0.069665, -0.138415, 0.173503, -0.101856, 0.235204, 0.069665, -0.138415, 0.324744, };
    double *bZeroYMat =  new double [BLOCK_SIZE*BLOCK_SIZE]{0.427563, -0.0457377, -0.0303162, -0.112822, -0.00623814, -0.198167, -0.0457377, 0.34276, -0.124948, 0.0932124, -0.202625, 0.122167, -0.0303162, -0.124948, 0.36952, 0.123022, 0.0919642, -0.274571, -0.112822, 0.0932124, 0.123022, 0.363345, -0.117247, -0.111981, -0.00623814, -0.202625, 0.0919642, -0.117247, 0.237397, -0.00975585, -0.198167, 0.122167, -0.274571, -0.111981, -0.00975585, 0.358075, };
    double *bZeroXVec =  new double [DIMENSION]{-0.324744, 0.138415, -0.069665, 0.0274035, -0.0185714, -0.235204, 0.101856, -0.0488784, 0.0221038, -0.173503, 0.0727448, -0.0379947, -0.125528, 0.0538021, };

    // current initial point
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
    // initial inputs
    std::vector<std::vector<double>> coefVectors =
            {
                    {
                            0.8267088500, -0.03762930000, 1.148176250, 0.3280084000,
                            0.7279020500, -0.2969960000, -0.4217697000, -0.2741615000,
                            -0.01951170000, 0.4880662500, -0.3687674000, 0.7530890500,
                            0.3540255000, -0.2344946000, 0.7331724500, 0.1567501000,
                            -0.07656680000, -0.08375230000, -0.5491424000, -0.5935686000,
                            0.3219790000, 0.4282622000, -0.3411265000, -0.1592065000,
                            0.1864248000, 0.9071691500, -0.04976520000, 0.7128398500, 1.009891000, -0.2623729000, 0.8567707500, 0.04297800000, -0.06063250000, -0.09147550000, 0.7973897000
                    }
            };
    double xVec[] = {-1.1847524076596173e-01,-1.4981342544862425e-03,-1.0317055199856009e-01,+1.7524849094959227e-03,-1.1137583709313288e-01,-9.2649792391757101e-03,+8.6261886761986376e-04,-4.2874455806629595e-03,+7.9012581990062648e-04,-1.0581945139187109e-01,-2.6789637899233012e-03,-1.0796166049029132e-01,-7.5830404877687725e-03,+4.5204889778688922e-03};


    int iterations =  1;
    double bStart = std::atof(argv[1]);
    double bEnd = std::atof(argv[2]);
    double bJumps = std::atof(argv[3]);
    int iteration;

    for (iteration = 0; iteration < iterations; ++iteration) {

        std::cout << "Iteration number: " << iteration + 1 << std::endl;

        double minPrimal = 300000000;
        double maxPrimal = -300000000;
        double minDual = 300000000;
        double maxDual = -300000000;
        bool initialPointEnabled = true;
        bool bZeroMatrices = false;
        startProblemSolving = chrono::steady_clock::now();

        for (double b = bStart; b <= bEnd; b+=bJumps) {

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

            if (initialPointEnabled) {
                sdpaProblem.setInitPoint(true);
                if(bZeroMatrices)
                    setInitialPoint(sdpaProblem, DIMENSION, BLOCK_SIZE, bZeroYMat, bZeroXMat, bZeroXVec);
                else
                    setInitialPoint(sdpaProblem, DIMENSION, BLOCK_SIZE, yMat, xMat, xVec);

            }

            buildMat(sdpaProblem);

            std::cout << "Current b value: " << b << std::endl;

            std:: vector<double> smallerProblem = extractSmallProblem(coefVectors[0], b);
            auto endCoefPrep = chrono::steady_clock::now();

            auto start = chrono::steady_clock::now();
            runProblem(sdpaProblem, 0, yMat, xMat, xVec, smallerProblem);
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

        endProblemSolving = chrono::steady_clock::now();

        std::cout << "Real minimum: " << minPrimal + coefVectors[0][34] << std::endl;
        std::cout << "min primal: " << minPrimal << std::endl;
        std::cout << "max primal: " << maxPrimal << std::endl;
        std::cout << "min dual: " << minDual << std::endl;
        std::cout << "max dual: " << maxDual << std::endl;
    }

    // print final statistics
    auto end_overall = chrono::steady_clock::now();
    auto problemsSolvingDuration = chrono::duration_cast<chrono::nanoseconds>(endProblemSolving - startProblemSolving).count();
    auto smallProblemsSolvedAmount = (bEnd - bStart) * iterations / bJumps;

    std::cout << "average sdpa solving time (nanos): " << sum / smallProblemsSolvedAmount << std::endl;
    std::cout << "average coef preparation time (nanos): " << sumCoefPrep / smallProblemsSolvedAmount << std::endl;
    std::cout << "average small problem solving time (nanos): " << (sum + sumCoefPrep) /  smallProblemsSolvedAmount<< std::endl;
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
                double yMat[BLOCK_SIZE][BLOCK_SIZE],
                double xMat[BLOCK_SIZE][BLOCK_SIZE],
                double xVec[DIMENSION],
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
//    fprintf(stdout, "iterations : %d\n", sdpaProblem.getIteration());


    sdpaProblem.terminate();
}


void setInitialPoint(SDPA &problem, int dim, int blockSize, double yMat[BLOCK_SIZE][BLOCK_SIZE],
                     double xMat[BLOCK_SIZE][BLOCK_SIZE], double xVec[DIMENSION]) {
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

void setInitialPoint(SDPA &problem, int dim, int blockSize, double *yMat, double *xMat, double *xVec) {
    for (int i = 0; i < blockSize; ++i) {
        for (int j = i; j < blockSize; ++j) {
            if (yMat[i + blockSize * j] != 0) {
                problem.inputInitYMat(1, i + 1, j + 1, yMat[i + blockSize * j]);
            }
        }
    }

    for (int i = 0; i < blockSize; ++i) {
        for (int j = i; j < blockSize; ++j) {
            if (xMat[i + blockSize * j] != 0) {
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

void printMatrices(SDPA &sdpaProblem){
    cout<<std::endl;
    cout<<std::endl;
    auto resultxVec = sdpaProblem.getResultXVec();
    for ( int i = 0; i < DIMENSION; i++)
        cout << *(resultxVec + i) << ", ";
    cout<<std::endl;
    cout<<std::endl;
    auto resultxMat = sdpaProblem.getResultXMat(1);
    for ( int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++)
        cout << *(resultxMat + i) << ", ";
    cout<<std::endl;
    cout<<std::endl;
    auto resultyMat = sdpaProblem.getResultYMat(1);
    for ( int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++)
        cout << *(resultyMat + i) << ", ";
    cout<<std::endl;
    cout<<std::endl;
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
