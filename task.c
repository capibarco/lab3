#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

double* matrixOld=0;
double* matrixNew=0;

double matrixCalc(int size)
{
	double errorNow = 0.0;
#pragma acc parallel loop independent collapse(2) vector vector_length(size) gang num_gangs(size) reduction(max:errorNow) present(matrixOld[0:size*size], matrixNew[0:size*size])
	for (int i = 1; i < size - 1; i++)
	{
		for (int j = 1; j < size - 1; j++)
		{
			matrixNew[i * size + j] = 0.25 * (
				matrixOld[i * size + j - 1] +
				matrixOld[(i - 1) * size + j] +
				matrixOld[(i + 1) * size + j] +
				matrixOld[i * size + j + 1]);
			errorNow = fmax(errorNow, matrixNew[i * size + j] - matrixOld[i * size + j]);
		}
	}
	return errorNow;
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

int main(int argc, char** argv)
{
	int cornerUL = 10;
	int cornerUR = 20;
	int cornerBR = 30;
	int cornerBL = 20;
	
	char* eptr;
	const double maxError = strtod((argv[1]), &eptr);
	const int size = atoi(argv[2]);
	const int maxIteration = atoi(argv[3]);

	int totalSize = size * size;
	
	matrixOld = (double*)calloc(totalSize , sizeof(double));
	matrixNew = (double*)calloc(totalSize , sizeof(double));

	const double fraction = 10.0 / (size - 1);
	double errorNow = 1.0;
	int iterNow = 0;
	clock_t begin = clock();	
#pragma acc enter data create(matrixOld[0:totalSize], matrixNew[0:totalSize]) copyin(errorNow)
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
			iterNow++;
			errorNow = matrixCalc(size);
			matrixSwap(totalSize);
	}
	
#pragma acc exit data delete(matrixOld[0:totalSize], matrixNew[0:totalSize])

	clock_t end = clock();	
	free(matrixOld);
   	free(matrixNew);
	printf("iterations = %d, error = %lf, time = %lf\n", iterNow, errorNow, (double)(end - begin) / CLOCKS_PER_SEC);
		
	return 0;
}
