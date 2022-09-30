
#include <cstdio>
#include <cstdlib>
#include <sdpa_call.h>
#include <chrono>

#define DIMENSION 34
#define BLOCK_SIZE 10

using std::cout;
using std::endl;

void printVector(const double *coefficients, int dimension);

void buildMat(SDPA &problem);

void runProblem(SDPA &problem, std::vector<double> &coef);

void putCoefficients(SDPA &problem, const std::vector<double> &coef);

void setInitialPoint(SDPA &problem, int dim, int blockSize, double yMat[BLOCK_SIZE][BLOCK_SIZE], double xMat[BLOCK_SIZE][BLOCK_SIZE], double xVec[DIMENSION]);
int i = 0;

int main() {
    // initialize general variables
    auto start_overall = chrono::steady_clock::now();
    long sum = 0;

    // initial inputs
    std::vector<std::vector<double>> coefVectors =
            {
                    {0.1487731000, 0.007404400000, 1.291135200, -0.05259190000, 0.3300128000, -0.05955900000, -0.2089143000, 0.4441555000, 0.2854625000, 0.9635312000, -0.7013847000, 0.772196000, 0.1052276000, 0.1593876000, 0.0317093500, -0.1466180000, -0.09105170000, 0.2566314000, 0.06830060000, -0.2846121000, 0.06115390000, 0.6638104000, 0.2830806000, 0.7631010000, 0.07735970000, 1.043273850, 0.4710202000, 1.168491000, -0.1290381000, -0.8049054000, 1.090042500, 0.05250310000, 0.1969399000, 0.3315520000}
            };

    // current initial point
    double xMat[BLOCK_SIZE][BLOCK_SIZE] = { {+6.3932582153893511e-02,-6.3803571001887095e-03,-5.6887590186935960e-03,+4.5038767512170178e-03,+6.2807030341918216e-02,-2.3688146692632185e-03,+2.4012254290552864e-03,+5.8920728079351857e-02,-1.2673207184824241e-03,+5.5755975011998929e-02 },
                      {-6.3803571001887095e-03,+6.2807030341918230e-02,-2.3688146692632185e-03,+2.4012254290552864e-03,-1.3870492761470832e-04,+9.7083572080443163e-04,+2.2848078339443834e-03,-2.5100248920644883e-03,+7.1635874364571928e-04,-1.6616456312189797e-03 },
                      {-5.6887590186935960e-03,-2.3688146692632185e-03,+5.8920728079351857e-02,-1.2673207184824241e-03,+9.7083572080443163e-04,-2.5100248920644883e-03,+7.1635874364571928e-04,-3.1398297840381515e-03,+1.4781821789800194e-03,-1.1801586088156569e-03 },
                      {+4.5038767512170178e-03,+2.4012254290552864e-03,-1.2673207184824241e-03,+5.5755975011998929e-02,+2.2848078339443834e-03,+7.1635874364571928e-04,-1.6616456312189797e-03,+1.4781821789800194e-03,-1.1801586088156569e-03,+1.8096091283639408e-03 },
                      {+6.2807030341918216e-02,-1.3870492761470832e-04,+9.7083572080443163e-04,+2.2848078339443834e-03,+7.6234433606577293e-02,+1.9143442623144337e-03,+8.8266839809531504e-03,+6.2246449661917794e-02,-9.7959211703423912e-04,+6.1161650485366724e-02 },
                      {-2.3688146692632185e-03,+9.7083572080443163e-04,-2.5100248920644883e-03,+7.1635874364571928e-04,+1.9143442623144337e-03,+6.2246449661917780e-02,-9.7959211703423912e-04,+4.7028163872208494e-03,+3.9427367127197943e-03,-2.2260589605623812e-04 },
                      {+2.4012254290552864e-03,+2.2848078339443834e-03,+7.1635874364571928e-04,-1.6616456312189797e-03,+8.8266839809531504e-03,-9.7959211703423912e-04,+6.1161650485366724e-02,+3.9427367127197943e-03,-2.2260589605623812e-04,+3.5572774920061140e-03 },
                      {+5.8920728079351857e-02,-2.5100248920644883e-03,-3.1398297840381515e-03,+1.4781821789800194e-03,+6.2246449661917794e-02,+4.7028163872208494e-03,+3.9427367127197943e-03,+7.1856523845314990e-02,-9.9215834773707431e-04,+5.9684449092670434e-02 },
                      {-1.2673207184824241e-03,+7.1635874364571928e-04,+1.4781821789800194e-03,-1.1801586088156569e-03,-9.7959211703423912e-04,+3.9427367127197943e-03,-2.2260589605623812e-04,-9.9215834773707431e-04,+5.9684449092670427e-02,+6.1373931857342995e-05 },
                      {+5.5755975011998929e-02,-1.6616456312189797e-03,-1.1801586088156569e-03,+1.8096091283639408e-03,+6.1161650485366724e-02,-2.2260589605623812e-04,+3.5572774920061140e-03,+5.9684449092670434e-02,+6.1373931857342995e-05,+6.6823895047766299e-02 }   };
    double yMat[BLOCK_SIZE][BLOCK_SIZE] =    { {+4.7213044902544632e-01,+3.3129602499999945e-02,+3.1726846250000003e-02,-2.0417685000000012e-02,-1.4610003815626954e-01,+2.6642788454662905e-02,+1.0768488265778269e-02,-1.2536758581991936e-01,+3.2697544636972162e-03,-1.1220851875341209e-01 },
                                               {+3.3129602499999945e-02,+3.0190449836343181e-01,+1.7532702795337109e-02,-1.9911180765778261e-02,-1.5424043749999991e-02,-4.0308748032090638e-03,-1.5861388862907860e-02,+2.6582669339250491e-03,-5.2953206457819020e-03,-3.5284605772610165e-03 },
                                               {+3.1726846250000003e-02,+1.7532702795337109e-02,+3.2449666169073099e-01,+8.1888217863027944e-03,-2.3547636446790927e-02,+1.8952053066074941e-02,-4.3546349337745174e-03,+8.2474737499999857e-03,-1.1731204789550340e-02,-4.0887120100924872e-03 },
                                               {-2.0417685000000012e-02,-1.9911180765778261e-02,+8.1888217863027944e-03,+3.4275971355771695e-01,-1.5324298870921561e-03,-6.1818894204435700e-03,+1.3206596827261011e-02,+4.9311247895503282e-03,+8.5561907600925080e-03,+1.4754037500000093e-03 },
                                               {-1.4610003815626954e-01,-1.5424043749999991e-02,-2.3547636446790927e-02,-1.5324298870921561e-03,+4.3309685322544644e-01,-9.3605487499999942e-03,-3.5530583749999997e-02,-1.1226098783556365e-01,+3.0774264820862228e-03,-1.2102330550014748e-01 },
                                               {+2.6642788454662905e-02,-4.0308748032090638e-03,+1.8952053066074941e-02,-6.1818894204435700e-03,-9.3605487499999942e-03,+2.9962194072202042e-01,+7.7912960179137899e-03,-2.7825338750000005e-02,-2.8667547304239000e-02,+7.9778722439694562e-03 },
                                               {+1.0768488265778269e-02,-1.5861388862907860e-02,-4.3546349337745174e-03,+1.3206596827261011e-02,-3.5530583749999997e-02,+7.7912960179137899e-03,+3.0900895305118770e-01,-2.3799164457610220e-03,+1.4532540060305393e-03,+1.7228512500000073e-03 },
                                               {-1.2536758581991936e-01,+2.6582669339250491e-03,+8.2474737499999857e-03,+4.9311247895503282e-03,-1.1226098783556365e-01,-2.7825338750000005e-02,-2.3799164457610220e-03,+4.3294656072544613e-01,+3.4795575000000005e-03,-1.2589008823944761e-01 },
                                               {+3.2697544636972162e-03,-5.2953206457819020e-03,-1.1731204789550340e-02,+8.5561907600925080e-03,+3.0774264820862228e-03,-2.8667547304239000e-02,+1.4532540060305393e-03,+3.4795575000000005e-03,+3.1926256852978829e-01,-5.2922024999999999e-03 },
                                               {-1.1220851875341209e-01,-3.5284605772610165e-03,-4.0887120100924872e-03,+1.4754037500000093e-03,-1.2102330550014748e-01,+7.9778722439694562e-03,+1.7228512500000073e-03,-1.2589008823944761e-01,-5.2922024999999999e-03,+4.4294495822544633e-01 }   };
    double xVec[] = {-6.6823895047766299e-02,-6.1373931857342995e-05,-5.9684449092670434e-02,+9.9215834773707431e-04,-7.1856523845314976e-02,-3.5572774920061140e-03,+2.2260589605623812e-04,-3.9427367127197943e-03,-4.7028163872208494e-03,-6.1161650485366724e-02,+9.7959211703423912e-04,-6.2246449661917794e-02,-8.8266839809531504e-03,-1.9143442623144337e-03,-7.6234433606577293e-02,-1.8096091283639408e-03,+1.1801586088156569e-03,-1.4781821789800194e-03,+3.1398297840381515e-03,+1.6616456312189797e-03,-7.1635874364571928e-04,+2.5100248920644883e-03,-2.2848078339443834e-03,-9.7083572080443163e-04,+1.3870492761470832e-04,-5.5755975011998929e-02,+1.2673207184824241e-03,-5.8920728079351857e-02,-2.4012254290552864e-03,+2.3688146692632185e-03,-6.2807030341918216e-02,-4.5038767512170178e-03,+5.6887590186935960e-03,+6.3803571001887095e-03};

    int iterations =  3;
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
        bool initialPointEnabled = false;

        // set initial point if not first iteration
        if (initialPointEnabled && iteration > 0) {
            sdpaProblem.setInitPoint(true);
            setInitialPoint(sdpaProblem, DIMENSION, BLOCK_SIZE, yMat, xMat, xVec);
        }

        std::cout << "Iteration number: " << iteration + 1 << std::endl;

        // run SPDA and measure time
        auto start = chrono::steady_clock::now();
        runProblem(sdpaProblem, coefVectors[0]);
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
                std::vector<double> &coef) {

    putCoefficients(sdpaProblem, coef);
    buildMat(sdpaProblem);

    sdpaProblem.initializeUpperTriangle();
    sdpaProblem.initializeSolve();
    sdpaProblem.solve();

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

void buildMat(SDPA &problem) {
    problem.inputElement(0,1,1,1,-1  );
    problem.inputElement(1,1,10,10,-1);
    problem.inputElement(1,1,1,1,1   );
    problem.inputElement(2,1,10,9,-1 );
    problem.inputElement(3,1,10,8,-1 );
    problem.inputElement(3,1,9,9,-1  );
    problem.inputElement(3,1,1,1,2   );
    problem.inputElement(4,1,9,8,-1  );
    problem.inputElement(5,1,8,8,-1  );
    problem.inputElement(5,1,1,1,1   );
    problem.inputElement(6,1,10,7,-1 );
    problem.inputElement(7,1,10,6,-1 );
    problem.inputElement(7,1,9,7,-1  );
    problem.inputElement(8,1,9,6,-1  );
    problem.inputElement(8,1,8,7,-1  );
    problem.inputElement(9,1,8,6,-1  );
    problem.inputElement(10,1,10,5,-1);
    problem.inputElement(10,1,7,7,-1 );
    problem.inputElement(10,1,1,1,2  );
    problem.inputElement(11,1,9,5,-1 );
    problem.inputElement(11,1,7,6,-1 );
    problem.inputElement(12,1,8,5,-1 );
    problem.inputElement(12,1,6,6,-1 );
    problem.inputElement(12,1,1,1,2  );
    problem.inputElement(13,1,7,5,-1 );
    problem.inputElement(14,1,6,5,-1 );
    problem.inputElement(15,1,5,5,-1 );
    problem.inputElement(15,1,1,1,1  );
    problem.inputElement(16,1,10,4,-1);
    problem.inputElement(17,1,10,3,-1);
    problem.inputElement(17,1,9,4,-1 );
    problem.inputElement(18,1,8,4,-1 );
    problem.inputElement(18,1,9,3,-1 );
    problem.inputElement(19,1,8,3,-1 );
    problem.inputElement(20,1,10,2,-1);
    problem.inputElement(20,1,7,4,-1 );
    problem.inputElement(21,1,6,4,-1 );
    problem.inputElement(21,1,7,3,-1 );
    problem.inputElement(21,1,9,2,-1 );
    problem.inputElement(22,1,6,3,-1 );
    problem.inputElement(22,1,8,2,-1 );
    problem.inputElement(23,1,5,4,-1 );
    problem.inputElement(23,1,7,2,-1 );
    problem.inputElement(24,1,5,3,-1 );
    problem.inputElement(24,1,6,2,-1 );
    problem.inputElement(25,1,5,2,-1 );
    problem.inputElement(26,1,10,1,-1);
    problem.inputElement(26,1,4,4,-1);
    problem.inputElement(26,1,1,1,2);
    problem.inputElement(27,1,4,3,-1);
    problem.inputElement(27,1,9,1,-1);
    problem.inputElement(28,1,3,3,-1);
    problem.inputElement(28,1,8,1,-1);
    problem.inputElement(28,1,1,1,2 );
    problem.inputElement(29,1,4,2,-1);
    problem.inputElement(29,1,7,1,-1);
    problem.inputElement(30,1,3,2,-1);
    problem.inputElement(30,1,6,1,-1);
    problem.inputElement(31,1,2,2,-1);
    problem.inputElement(31,1,5,1,-1);
    problem.inputElement(31,1,1,1,2 );
    problem.inputElement(32,1,4,1,-1);
    problem.inputElement(33,1,3,1,-1);
    problem.inputElement(34,1,2,1,-1);
}

void printVector(const double *coefficients, int dimension) {
    for (int i = 0; i < dimension; ++i) {
        fprintf(stdout, "y%d = %.17g\n", i + 1, coefficients[i]);
    }
}
