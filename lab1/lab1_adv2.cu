#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>

using namespace std;

__global__ void kernel(int* v1, int* v2, int* ans, long long n) { //отличное от C++ (__global__)
	long long i, idx = blockDim.x * blockIdx.x + threadIdx.x;			// Абсолютный номер потока 
	//blockDim (кол-во потоков в блоке) и blockIdx (номер блока в thread) - глобальные константы в CUDA, по осям x,y,z и т.д
	//threadIds (номер thread-a) 
	long long offset = blockDim.x * gridDim.x;						// Общее кол-во потоков
	for (i = idx; i < n; i += offset) // Для всех требование - внутри цикла for()
	// Данных в тестах больше, чем поток, который можно выделить => нужно завернуть в цикл
	// offset - суммарное кол-во выделенных потоков на обработку
	// без него будет падать
		ans[i] = ((v1[i] * v1[i]) + 2 * (v1[i] * v2[i]) + (v2[i] * v2[i]));
}

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	long long n;
	cin >> n;
	int* vector1 = (int*)malloc(sizeof(int) * n); //выделение массива 1
	int* vector2 = (int*)malloc(sizeof(int) * n); //выделение массива 2
	int* answer = (int*)malloc(sizeof(int) * n); //выделение массива 2
	
	for (long long i = 0; i < n; i++)
		vector1[i] = rand()%10+1; //заполнение массива 1

	for (long long i = 0; i < n; i++)
		cout << vector1[i] << ' ';
	cout << '\n';
	cout << '\n';

	for (long long i = 0; i < n; i++)
		vector2[i] = rand()%10+1; //заполнение массива 2

	for (long long i = 0; i < n; i++)
		cout << vector2[i] << ' ';
	cout << '\n';
	cout << '\n';

	int* dev_arr1;
	cudaMalloc(&dev_arr1, sizeof(int) * n); //выделение массива на устройстве 
	cudaMemcpy(dev_arr1, vector1, sizeof(int) * n, cudaMemcpyHostToDevice);

	int* dev_arr2;
	cudaMalloc(&dev_arr2, sizeof(int) * n); //выделение массива на устройстве 
	cudaMemcpy(dev_arr2, vector2, sizeof(int) * n, cudaMemcpyHostToDevice);

	int* result;
	cudaMalloc(&result, sizeof(int) * n); //выделение массива на устройстве 
	cudaMemcpy(result, answer, sizeof(int) * n, cudaMemcpyHostToDevice);

	kernel << <256, 256 >> > (dev_arr1, dev_arr2, result, n); //отличное от C++ (<<<>>>), стандартная функция
	// Многопоточное
	// 256 блоков и 256 потоков(Thread)

	cudaMemcpy(vector1, dev_arr1, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaFree(dev_arr1);

	cudaMemcpy(vector2, dev_arr2, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaFree(dev_arr2);

	cudaMemcpy(answer, result, sizeof(int) * n, cudaMemcpyDeviceToHost);
	cudaFree(result);

	cout.precision(10);
	cout.setf(ios::scientific);
	for (long long i = 0; i < n; i++)
		cout << answer[i] << ' ';
	cout << endl;
	free(vector1);
	free(vector2);
	free(answer);
	return 0;
}
