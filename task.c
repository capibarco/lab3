#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>
/*
double* matrixOld = 0;
double* matrixNew = 0;
double* matrixTmp = 0;

void matrixCalc(int size)
{
#pragma acc parallel loop independent collapse(2) vector vector_length(size) gang num_gangs(size) present(matrixOld[0:size*size], matrixNew[0:size*size])
	for (int i = 1; i < size - 1; i++)
	{
		for (int j = 1; j < size - 1; j++)
		{
			matrixNew[i * size + j] = 0.25 * (
				matrixOld[i * size + j - 1] +
				matrixOld[(i - 1) * size + j] +
				matrixOld[(i + 1) * size + j] +
				matrixOld[i * size + j + 1]);
		}
	}
}

void matrixSwap(int totalSize)
{
#pragma acc data present(matrixOld[0:totalSize], matrixNew[0:totalSize])
	{
		double* temp = matrixOld;
		matrixOld = matrixNew;
		matrixNew = temp;
	}
}
*/
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

	double* matrixOld = (double*)calloc(totalSize, sizeof(double));
	double* matrixNew = (double*)calloc(totalSize, sizeof(double));
	double* matrixTmp = (double*)malloc(totalSize * sizeof(double));
	const double fraction = 10.0 / (size - 1);
	double errorNow = 1.0;
	int iterNow = 0;
	int result = 0;
	const double minus = -1;
	int iters_up = 0;
	clock_t begin = clock();
	for (int i = 0; i < size; i++)
	{
		matrixOld[i] = cornerUL + i * fraction;
		matrixOld[i * size] = cornerUL + i * fraction;
		matrixOld[size * i + size - 1] = cornerUR + i * fraction;
		matrixOld[size * (size - 1) + i] = cornerUR + i * fraction;

		matrixNew[i] = matrixOld[i];
		matrixNew[i * size] = matrixOld[i * size];
		matrixNew[size * i + size - 1] = matrixOld[size * i + size - 1];
		matrixNew[size * (size - 1) + i] = matrixOld[size * (size - 1) + i];
	}
#pragma acc enter data copyin(matrixOld[0:totalSize], matrixNew[0:totalSize], matrixTmp[0:totalSize])
	while (errorNow > maxError && iterNow < maxIteration)
	{
#pragma acc parallel loop collapse(2)  present(matrixOld[0:totalSize], matrixNew[0:totalSize]) vector_length(128) async
		for (int i = 1; i < size - 1; i++)
		{
			for (int j = 1; j < size - 1; j++)
			{
				matrixNew[i * size + j] = 0.25 * (
					matrixOld[i * size + j - 1] +
					matrixOld[(i - 1) * size + j] +
					matrixOld[(i + 1) * size + j] +
					matrixOld[i * size + j + 1]);
			}
		}
		
		double* temp = matrixOld;
		matrixOld = matrixNew;
		matrixNew = temp;
		#ifdef OPENACC__
			acc_attach((void**)matrixOld);
			acc_attach((void**)matrixNew);
		#endif
		 if (iters_up >= 2 && iterNow < maxIteration) 
		 {
#pragma acc data present(matrixOld[0:totalSize], matrixNew[0:totalSize], matrixTmp[0:totalSize]) wait
		{
#pragma acc host_data use_device(matrixNew, matrixOld, matrixTmp)
			{
				stat = cublasDcopy(handle, totalSize, matrixOld, 1, matrixTmp, 1);
				if (stat != CUBLAS_STATUS_SUCCESS)
				{
					printf("cublasDcopy error\n");
					cublasDestroy(handle);
					return EXIT_FAILURE;
				}

				stat = cublasDaxpy(handle, totalSize, &minus, matrixNew, 1, matrixTmp, 1);
				if (stat != CUBLAS_STATUS_SUCCESS)
				{
					printf("cublasDaxpy error\n");
					cublasDestroy(handle);
					return EXIT_FAILURE;
				}

				stat = cublasIdamax(handle, totalSize, matrixTmp, 1, &result);
				if (stat != CUBLAS_STATUS_SUCCESS)
				{
					printf("cublasIdamax error\n");
					cublasDestroy(handle);
					return EXIT_FAILURE;
				}
			}
		}
		
#pragma acc update self(matrixTmp[result-1])
		errorNow = fabs(matrixTmp[result-1]);	
			 iters_up=-1;
		 }
		iters_up++;
		iterNow++;
	}

#pragma acc exit data delete(matrixNew[0:totalSize])
	clock_t end = clock();
	cublasDestroy(handle);
	free(matrixOld);
	free(matrixNew);

	printf("iterations = %d, error = %lf, time = %lf\n", iterNow, errorNow, (double)(end - begin) / CLOCKS_PER_SEC);

	return 0;
}
