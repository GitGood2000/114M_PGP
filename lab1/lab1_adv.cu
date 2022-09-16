#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

//32x32
#define NTHREADS_X 32
#define NTHREADS_Y 32
#define THREADS_PER_BLOCK NTHREADS_X * NTHREADS_Y

__global__ void kernel(int* m1, int* m2, int* ans, int a1, int a2, int b1, int b2) { //отличное от C++ (__global__)
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;			// Абсолютный номер потока 
	//blockDim (кол-во потоков в блоке) и blockIdx (номер блока в thread) - глобальные константы в CUDA, по осям x,y,z и т.д
	//threadIds (номер thread-a) 
	//int offsetx = blockDim.x * gridDim.x;
	//int offsety = blockDim.y * gridDim.y;						// Общее кол-во потоков
	int i, tmp_sum = 0;

	int A_start = a2 * idy;
	int B_start = idx;

	if (idx >= b2 || idy >= a1) // Предотвратить заход за границу блоков
		return;

	for (i = 0; i < a2; i += 1) // Для всех требование - внутри цикла for()
	// Данных в тестах больше, чем поток, который можно выделить => нужно завернуть в цикл
	// offset - суммарное кол-во выделенных потоков на обработку
	// без него будет падать
		tmp_sum += m1[A_start + i] * m2[i * b2 + B_start];

	ans[idy * b2 + idx] = tmp_sum;
}

int main() {
	srand((unsigned)time(0));

	long int a1, a2, b1, b2;
	scanf("%ld", &a1);
	scanf("%ld", &a2);
	scanf("%ld", &b1);
	scanf("%ld", &b2);

	if (a2 != b1)
	{
		printf("Number of columns in Matrix A should be equals to number of lines in Matrix B\n");
		return EXIT_FAILURE;
	}

	size_t  size_a = a1 * a2 * sizeof(int);
	size_t  size_b = b1 * b2 * sizeof(int);
	size_t  size_c = a1 * b2 * sizeof(int);

	int* matrix1 = (int*)malloc(size_a); //выделение массива 1
	int* matrix2 = (int*)malloc(size_b); //выделение массива 2
	int* answer = (int*)malloc(size_c); //выделение массива ответа

	for (long int i = 0; i < a1; i++) {
		for (long int j = 0; j < a2; j++) {
			matrix1[i * a2 + j] = (rand() % 10); //заполнение массива 1
		}
	}

	/*for (long int i = 0; i < a1; i++) {
		for (long int j = 0; j < a2; j++) {
			printf("%ld ", matrix1[i * a2 + j]);
		}
		printf("\n");
	}
	printf("\n");*/

	srand((unsigned)time(0));

	for (long int i = 0; i < b1; i++) {
		for (long int j = 0; j < b2; j++) {
			matrix2[i * b2 + j] = (rand() % 10) ; //заполнение массива 2
		}
	}

	/*for (long int i = 0; i < b1; i++) {
		for (long int j = 0; j < b2; j++) {
			printf("%ld ", matrix2[i * b2 + j]);
		}
		printf("\n");
	}
	printf("\n"); */

	//printf("ready to cuda malloc \n");

	int* dev_arr1;
	cudaMalloc(&dev_arr1, size_a); //выделение массива на устройстве 
	cudaMemcpy(dev_arr1, matrix1, size_a, cudaMemcpyHostToDevice);

	//printf("a complete \n");

	int* dev_arr2;
	cudaMalloc(&dev_arr2, size_b); //выделение массива на устройстве 
	cudaMemcpy(dev_arr2, matrix2, size_b, cudaMemcpyHostToDevice);

	//printf("b complete \n");

	int* result;
	cudaMalloc(&result, size_c); //выделение массива на устройстве 
	cudaMemcpy(result, answer, size_c, cudaMemcpyHostToDevice);

	//printf("c complete \n");

	// определяем количество блоков и потоков по размеру матрицы результатов
	dim3 tblock = dim3(
		(int)ceil((double)b2 / NTHREADS_X),
		(int)ceil((double)a1 / NTHREADS_Y),
		1
	);

	dim3 tthreads = dim3(
		NTHREADS_X,
		NTHREADS_Y,
		1
	);

	//printf("tblock.x: %d tblock.y: %d tblock.z: %d\n", tblock.x, tblock.y, tblock.z);
	//printf("tthreads.x: %d tthreads.y: %d\n", tthreads.x, tthreads.y);

	kernel << <tblock, tthreads >> > (dev_arr1, dev_arr2, result, a1, a2, b1, b2); //отличное от C++ (<<<>>>), стандартная функция
	//kernel << <256, 256 >> > (dev_arr1, dev_arr2, result, a1, a2, b1, b2); //отличное от C++ (<<<>>>), стандартная функция
	// Многопоточное

	cudaFree(dev_arr1);
	cudaFree(dev_arr2);

	cudaMemcpy(answer, result, size_c, cudaMemcpyDeviceToHost);
	cudaFree(result);

	//cout.precision(10);
	//cout.setf(ios::scientific);
	for (long int i = 0; i < a1; i++) {
		for (long int j = 0; j < b2; j++) {
			printf("%ld ", answer[i*b2 + j]);
		}
		printf("\n");
	}
	printf("\n");
	free(matrix1);
	free(matrix2);
	free(answer);
	return 0;
}
