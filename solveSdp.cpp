
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

int main() {
    // initialize general variables
    auto start_overall = chrono::steady_clock::now();
    long sum = 0;

    // initial inputs
    std::vector<std::vector<double>> coefVectors =
            {
                    {
                            -35.72303600, 2.895776000, -28.68785800, 9.272296000,
                            -39.41445700, -6.033732000, -1.080024000, 14.36109200,
                            3.748776000, -45.12376200, 3.010502000, -29.94402400,
                            0.7073560000, -4.064496000, -37.22224400, -0.5941640000,
                            -2.382526000, -6.699726000, -2.168924000, -4.103466000,
                            -3.166836000, -28.99672000, 18.84646400, -5.931412000,
                            0.07014600000, -39.97219400, 11.52328000, -22.74463400,
                            -7.755768000, -2.999454000, -27.97190500, -6.329940000,
                            0.5362840000, 8.421070000, -35.47082700
                    },
                    {
                            -35.72307200, 2.895530600, -28.68831100, 9.272177390,
                            -39.41446540, -6.033316800, -1.078391660, 14.36203297,
                            3.748880980, -45.12544335, 3.007599165, -29.94458565,
                            0.7101482200, -4.063035330, -37.22387212, -0.5937446000,
                            -2.380812145, -6.698553960, -2.168786540, -4.105331940,
                            -3.167857785, -28.99731085, 18.84609780, -5.931166420,
                            0.07386223500, -39.97308190, 11.52276039, -22.74503445,
                            -7.760910655, -3.002643130, -27.97178215, -6.331883220,
                            0.5349662800, 8.418509620, -35.47159985
                    },
                    {
                            -35.72479010, 2.892915105, -28.69027632, 9.271439120,
                            -39.41460230, -6.032504380, -1.074855200, 14.36448341,
                            3.749704660, -45.12830825, 3.004801705, -29.94660625,
                            0.7108028600, -4.060683970, -37.22498768, -0.5902669450,
                            -2.374534080, -6.694808550, -2.167761620, -4.105627745,
                            -3.171268860, -29.00024578, 18.84877560, -5.928687620,
                            0.07428641500, -39.97390492, 11.51979860, -22.74668847,
                            -7.761664900, -3.003994910, -27.97106762, -6.332831290,
                            0.5339764000, 8.418366110, -35.47172752
                    },
                    {
                            -35.72511410, 2.892902505, -28.69040245, 9.271436670,
                            -39.41461455, -6.030785380, -1.076580375, 14.36478346,
                            3.749362710, -45.13125792, 3.009453860, -29.94912275,
                            0.7125791600, -4.062501190, -37.22533365, -0.5908627450,
                            -2.372779865, -6.694890065, -2.167418270, -4.102333620,
                            -3.177536750, -28.99512040, 18.84361414, -5.922212440,
                            0.07605713500, -39.97555222, 11.52139545, -22.74936142,
                            -7.756446015, -3.012391725, -27.97475255, -6.334094053,
                            0.5377189150, 8.421997990, -35.47318295
                    },
                    {
                            -35.72580842, 2.893329375, -28.68807812, 9.270702000,
                            -39.41667118, -6.029720840, -1.077824595, 14.36323320,
                            3.750940890, -45.13375552, 3.010799150, -29.94582925,
                            0.7141810200, -4.063881010, -37.22690575, -0.5918456000,
                            -2.373834760, -6.692781360, -2.165082745, -4.104109760,
                            -3.176367870, -28.99166290, 18.84407440, -5.925924815,
                            0.07225073500, -39.97751795, 11.52093232, -22.74723999,
                            -7.756996135, -3.015932085, -27.97949108, -6.335239167,
                            0.5361378650, 8.419050790, -35.47412545
                    },
                    {
                            -35.72766602, 2.892226015, -28.68824628, 9.270700720,
                            -39.41667118, -6.030130290, -1.074446475, 14.36427208,
                            3.750944950, -45.13729072, 3.010141650, -29.94748170,
                            0.7137938950, -4.060572110, -37.22856632, -0.5901431500,
                            -2.371695670, -6.692294265, -2.165080850, -4.107641665,
                            -3.178896185, -28.99320596, 18.84527410, -5.920876610,
                            0.06873401000, -39.98193355, 11.51898828, -22.74760376,
                            -7.755735360, -3.010504660, -27.98515905, -6.333394513,
                            0.5379077950, 8.415020580, -35.47630632
                    },
                    {
                            -35.72768125, 2.891976025, -28.68911085, 9.272037207,
                            -39.41710590, -6.030451650, -1.077465305, 14.36288448,
                            3.752965313, -45.13879318, 3.007751870, -29.95087160,
                            0.7158538950, -4.058149610, -37.22919132, -0.5902605400,
                            -2.372381915, -6.689378310, -2.166569540, -4.108525325,
                            -3.174483480, -28.99165232, 18.84979228, -5.918233280,
                            0.06644901000, -39.98194750, 11.52180958, -22.75001458,
                            -7.752114390, -3.011127115, -27.98861002, -6.332574290,
                            0.5359621450, 8.412529930, -35.47704890
                    },
                    {
                            -35.72818750, 2.891636275, -28.68940860, 9.271956420,
                            -39.41713450, -6.028449150, -1.077576355, 14.36309789,
                            3.752779133, -45.14163292, 3.009012060, -29.95137872,
                            0.7175537950, -4.058814290, -37.22955612, -0.5904000400,
                            -2.370358725, -6.688716880, -2.166077340, -4.110125925,
                            -3.179315030, -28.99049771, 18.85338516, -5.917927240,
                            0.06485607000, -39.98071735, 11.52251078, -22.75183580,
                            -7.754824880, -3.006331975, -27.98929650, -6.332403480,
                            0.5334275450, 8.414827600, -35.47780790
                    },
                    {
                            -35.72821720, 2.891827025, -28.68923142, 9.270404170,
                            -39.41910142, -6.028036587, -1.078607350, 14.35879734,
                            3.750388670, -45.14257560, 3.005398695, -29.95609210,
                            0.7141510800, -4.061237093, -37.23157662, -0.5901466150,
                            -2.370771900, -6.692065405, -2.169337065, -4.111996585,
                            -3.182994930, -28.99157823, 18.85206334, -5.920683980,
                            0.06576855500, -39.98147482, 11.52149840, -22.75142123,
                            -7.752846475, -3.004513340, -27.98761052, -6.331478130,
                            0.5348901950, 8.414423630, -35.47820390
                    },
                    {
                            -35.73033780, 2.892177005, -28.68771240, 9.270277630,
                            -39.41937865, -6.024324953, -1.081105610, 14.35763623,
                            3.751181210, -45.14434705, 3.007329135, -29.95660525,
                            0.7142800400, -4.061313253, -37.23157920, -0.5902018750,
                            -2.371011405, -6.692025285, -2.169248820, -4.114043500,
                            -3.182637000, -28.99094680, 18.85389507, -5.921775360,
                            0.06569575500, -39.97817800, 11.52122314, -22.75262039,
                            -7.755759255, -3.002929835, -27.98801352, -6.331435170,
                            0.5350799350, 8.416052530, -35.47948555
                    }
            };

    // current initial point
    double *yMat = new double[BLOCK_SIZE * BLOCK_SIZE];
    double *xMat = new double[BLOCK_SIZE * BLOCK_SIZE];
    double *xVec = new double[DIMENSION];

    int iterations =  coefVectors.size();
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
        bool initialPointEnabled = true;

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

        std::cout << "Iteration number: " << iteration + 1 << std::endl;

        // run SPDA and measure time
        auto start = chrono::steady_clock::now();
        runProblem(sdpaProblem, 0, yMat, xMat, xVec, copyToInitialPoint, coefVectors[iteration]);
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
