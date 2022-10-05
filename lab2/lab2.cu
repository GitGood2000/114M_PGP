#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>

using namespace std;

#define CSC(call)  													\
do {																\
	cudaError_t res = call;											\
	if (res != cudaSuccess) {										\
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		\
		exit(0);													\
	}																\
} while(0)

// __device__  - выполняется на gpu, вызывается с gpu
__device__ int find_perfect(int num) {
	if (num > 1) {
		//printf("find perfect initiated \n");
		int sum = 0;
		for (int i = 1; i < (num); i++) {
			if (num % i == 0)
				sum += i;
		}
		//printf("%d ", num);
		//printf("%d \n", sum);
		if (sum == num) {
			//printf("Calculation worked \n");
			return sum;
		}
		else return 0;
	}
	else return 0;
}

// __global__  - выполняется на gpu, вызывается с cpu
__global__ void kernel(int* arr, int* ans, int n) { //отличное от C++ (__global__)
	int i, idx = blockDim.x * blockIdx.x + threadIdx.x;			// Абсолютный номер потока 
	//blockDim (кол-во потоков в блоке) и blockIdx (номер блока в thread) - глобальные константы в CUDA, по осям x,y,z и т.д
	//threadIds (номер thread-a) 
	int offset = blockDim.x * gridDim.x;						// Общее кол-во потоков

	/*
	присвоить переменной, в которой будет накапливаться сумма делителей, 0.
	В цикле от 1 до половины текущего натурального числа
	пытаться разделить исследуемое число нацело на счетчик внутреннего цикла.
	Если делитель делит число нацело, то добавить его к переменной суммы делителей.
	Если сумма делителей равна исследуемому натуральному числу, то это число совершенно и следует вывести его на экран.
	*/

	//printf("Method launched \n");
	for (i = idx; i < n; i += offset) // Для всех требование - внутри цикла for()
	// Данных в тестах больше, чем поток, который можно выделить => нужно завернуть в цикл
	// offset - суммарное кол-во выделенных потоков на обработку
	// без него будет падать
		ans[i] = find_perfect(arr[i]);
}

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	int n;
	cin >> n;

	size_t size = n * sizeof(int);

	int* N_array = (int*)malloc(size); //выделение массива 1
	int* answer = (int*)malloc(size); //выделение массива 2
	for (int i = 0; i < n; i++)
		N_array[i] = i; //заполнение массива 1

	for (int i = 0; i < n; i++)
		answer[i] = 0; //заполнение массива 1
	

	int* dev_arr1;
	CSC(cudaMalloc(&dev_arr1, size)); //выделение массива на устройстве 
	CSC(cudaMemcpy(dev_arr1, N_array, size, cudaMemcpyHostToDevice));

	int* result;
	CSC(cudaMalloc(&result, size)); //выделение массива на устройстве 
	CSC(cudaMemcpy(result, answer, size, cudaMemcpyHostToDevice));

	kernel << <256, 256 >> > (dev_arr1, result, n); //отличное от C++ (<<<>>>), стандартная функция
	// Многопоточное
	// 256 блоков и 256 потоков(Thread)



	CSC(cudaMemcpy(answer, result, size, cudaMemcpyDeviceToHost));
	CSC(cudaFree(result));

	for (int i = 0; i < n; i++)
		if (answer[i] > 0) {
			cout << answer[i] << ' ';
		}
	cout << endl;
	free(N_array);
	free(answer);
	return 0;
}
