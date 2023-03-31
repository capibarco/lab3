#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>

double* matrixOld = 0;
double* matrixNew = 0;
double* matrixTmp = 0;
#define size_q size*size
#define at(arr, x, y) (arr[(x)*size+(y)])
void initArrays(double* mainArr, double* subArr, int size) {
memset(mainArr, 0, sizeof(double) * size_q);
for (int i = 0; i < size; i++)
{
at(mainArr, 0, i) = 10 / size * i + 10;
at(mainArr, i, 0) = 10 / size * i + 10;
at(mainArr, size - 1, i) = 10 / size * i + 20;
at(mainArr, i, size - 1) = 10 / size * i + 20;

at(subArr, 0, i) = 10 / size * i + 10;
at(subArr, i, 0) = 10 / size * i + 10;
at(subArr, size - 1, i) = 10 / size * i + 20;
at(subArr, i, size - 1) = 10 / size * i + 20;
}
memcpy(subArr, mainArr, sizeof(double) * size_q);
}

const int ITERS_BETWEEN_UPDATE = 75;


int main(int argc, char** argv)
{
	cublasStatus_t stat;
	cublasHandle_t handle;
	cublasCreate(&handle);

	int cornerUL = 10;
	int cornerUR = 20;
	int cornerBR = 30;
	int cornerBL = 20;

	char* eptr;
	const double maxError = strtod((argv[1]), &eptr);
	const int size = atoi(argv[2]);
	const int maxIteration = atoi(argv[3]);

	int totalSize = size * size;

	matrixOld = (double*)calloc(totalSize, sizeof(double));
	matrixNew = (double*)calloc(totalSize, sizeof(double));
	matrixTmp = (double*)malloc(totalSize * sizeof(double));
	double eps = maxError;
    int iterations = maxIteration;
 double error = 1;
    int iteration = 0;
    int iters_up = 0;
	  int max_idx = 0;
	const double minus = -1;
	initArrays(matrixOld, matrixNew, size);
	clock_t begin = clock();
#pragma acc enter data copyin(matrixNew[:size_q], matrixOld[:size_q], matrixTmp[:size_q])

    do 
	{
		#pragma acc parallel loop collapse(2) present(matrixNew[:size_q], matrixOld[:size_q]) vector_length(128) async
        for (int x = 1; x < size - 1; x++) 
            for (int y = 1; y < size - 1; y++) 
                at(matrixNew, x, y) = 0.25 * (at(matrixOld, x + 1, y) + at(matrixOld, x - 1, y) + at(matrixOld, x, y - 1) + at(matrixOld, x, y + 1));
        double* swap = matrixOld;
        matrixOld = matrixNew;
        matrixNew = swap;

		#ifdef OPENACC__
			acc_attach((void**)matrixOld);
			acc_attach((void**)matrixNew);
		#endif

        if (iters_up >= ITERS_BETWEEN_UPDATE && iteration < iterations) 
		{

			#pragma acc data present(matrixTmp[:size_q], matrixNew[:size_q], matrixOld[:size_q]) wait
			{
				#pragma acc host_data use_device(matrixNew, matrixOld, matrixTmp)
				{

					stat = cublasDcopy(handle, size_q, matrixOld, 1, matrixTmp, 1);
					
					stat = cublasDaxpy(handle, size_q, &minus, matrixNew, 1, matrixTmp, 1);
					

					stat = cublasIdamax(handle, size_q, matrixTmp, 1, &max_idx);
				}
			}

			#pragma acc update self(matrixTmp[max_idx-1])
            error = fabs(matrixTmp[max_idx - 1]);

            iters_up = -1;
        }
        iteration++;
        iters_up++;
    } while (iteration < iterations && error > eps);
#pragma acc exit data delete(matrixNew[:size_q]) copyout(matrixOld[:size_q])

	clock_t end = clock();
	cublasDestroy(handle);
	free(matrixOld);
	free(matrixNew);
	free(matrixTmp);
	printf("iterations = %d, error = %lf, time = %lf\n", iteration, error, (double)(end - begin) / CLOCKS_PER_SEC);

	return 0;
}
