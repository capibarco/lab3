#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cublas_v2.h>

/*
for (int i = 1; i < size-1; i++)
		{
			for (int j = 1; j < size-1; j++)
				printf("%lf\t",matrixNew[size * i + j]);
			printf("\n");				  
		}
		printf("\n");
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
	double* matrixTmp = (double*)calloc(totalSize, sizeof(double));
	
	const double fraction = 10.0 / (size - 1);
	double errorNow = 1.0;
	int iterNow = 0;
	int iters_up = 0;
	
	int result = 0;
	const double minus = -1;
	
	
	clock_t begin = clock();
	
	#pragma acc enter data create(matrixOld[0:totalSize], matrixNew[0:totalSize], matrixTmp[0:totalSize])
	#pragma acc parallel loop
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
	
	while (errorNow > maxError && iterNow < maxIteration)
	{
		
		#pragma acc parallel loop independent collapse(2) vector vector_length(size) gang num_gangs(size) present(matrixOld[0:totalSize], matrixNew[0:totalSize])
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
		
		#pragma acc host_data use_device(matrixNew, matrixOld, matrixTmp)
		{
			stat = cublasDcopy(handle, totalSize, matrixNew, 1, matrixTmp, 1);
			if (stat != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasDcopy error\n");
				cublasDestroy(handle);
				return EXIT_FAILURE;
			}

			stat = cublasDaxpy(handle, totalSize, &minus, matrixOld, 1, matrixTmp, 1);
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
			printf("O\n");
			for (int i = 1; i < size-1; i++)
		{
			for (int j = 1; j < size-1; j++)
				printf("%lf\t",matrixOld[size * i + j]);
			printf("\n");				  
		}
		printf("\n");
			printf("N\n");
			for (int i = 1; i < size-1; i++)
		{
			for (int j = 1; j < size-1; j++)
				printf("%lf\t",matrixNew[size * i + j]);
			printf("\n");				  
		}
		printf("\n");
			printf("T\n");
			for (int i = 1; i < size-1; i++)
		{
			for (int j = 1; j < size-1; j++)
				printf("%lf\t",matrixTmp[size * i + j]);
			printf("\n");				  
		}
		printf("\n");
			
		}

		
		#pragma acc update self(matrixTmp[result-1])
		errorNow = matrixTmp[result-1];	
		
		iterNow++;
		
	}

	#pragma acc exit data delete(matrixOld[0:totalSize], matrixNew[0:totalSize], matrixTmp[0:totalSize])
	clock_t end = clock();
	cublasDestroy(handle);
	free(matrixOld);
	free(matrixNew);
	free(matrixTmp);
	printf("iterations = %d, error = %lf, time = %lf\n", iterNow, errorNow, (double)(end - begin) / CLOCKS_PER_SEC);

	return 0;
}
