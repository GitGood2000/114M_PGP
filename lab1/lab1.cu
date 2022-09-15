#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include  <math.h>
#include <cstdlib>

using namespace std;

__global__ void kernel(double* v1, double* ans, long long n) { //отличное от C++ (__global__)
	long long i, idx = blockDim.x * blockIdx.x + threadIdx.x;			// Абсолютный номер потока 
	//blockDim (кол-во потоков в блоке) и blockIdx (номер блока в thread) - глобальные константы в CUDA, по осям x,y,z и т.д
	//threadIds (номер thread-a) 
	long long offset = blockDim.x * gridDim.x;						// Общее кол-во потоков
	for (i = idx; i < n; i += offset) // Для всех требование - внутри цикла for()
	// Данных в тестах больше, чем поток, который можно выделить => нужно завернуть в цикл
	// offset - суммарное кол-во выделенных потоков на обработку
	// без него будет падать
		ans[i] = exp(v1[i]);
}

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	long long n;
	cin >> n;
	double* vector1 = (double*)malloc(sizeof(double) * n); //выделение массива 1
	double* answer = (double*)malloc(sizeof(double) * n); //выделение массива ответа
	for (long long i = 0; i < n; i++)
		vector1[i] = i; //заполнение массива 1

	double* dev_arr1;
	cudaMalloc(&dev_arr1, sizeof(double) * n); //выделение массива на устройстве 
	cudaMemcpy(dev_arr1, vector1, sizeof(double) * n, cudaMemcpyHostToDevice);

	double* result;
	cudaMalloc(&result, sizeof(double) * n); //выделение массива на устройстве 
	cudaMemcpy(result, answer, sizeof(double) * n, cudaMemcpyHostToDevice);

	kernel << <256, 256 >> > (dev_arr1, result, n); //отличное от C++ (<<<>>>), стандартная функция
	// Многопоточное
	// 256 блоков и 256 потоков(Thread)

	cudaMemcpy(vector1, dev_arr1, sizeof(double) * n, cudaMemcpyDeviceToHost);
	cudaFree(dev_arr1);


	cudaMemcpy(answer, result, sizeof(double) * n, cudaMemcpyDeviceToHost);
	cudaFree(result);

	//cout.precision(10);
	//cout.setf(ios::scientific);
	for (long long i = 0; i < n; i++)
		cout << answer[i] << ' ';
	cout << endl;
	free(vector1);
	free(answer);
	return 0;
}