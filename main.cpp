#include <iostream>
#include <thread>
#include <vector>
#include "mpi.h"
#include <cmath>
#include <fstream>

using namespace std::chrono_literals;

int N = 1000;
double h = 1. / (N - 1);
double eps = 0.000001;
double k = N;

const double pi = 3.14159265359;

double f(double x, double y) {
	return (2 * sin(pi * y) + k * k * (1 - x) * x * sin(pi * y) + pi * pi * (1 - x) * x * sin(pi * y));
}
double u(double x, double y) {
	return ((1 - x) * x * sin(pi * y));
}
std::vector<double> GetExAnswer() {
	std::vector<double> answer(N * N);
	for (size_t i = 0; i < N; ++i)
		for (size_t j = 0; j < N; ++j)
			answer[i * N + j] = u(j * h, i * h);
	return answer;
}
void print(const std::vector<double>& vec,int N, int M, int id) {
	std::cout << "------------ "<<id<<" ----------" << "\n";
	for (int i = M-1; i >= 0; --i) {
		for (int j = 0; j < N; ++j)
			std::cout << vec[i * N + j] << "  ";
		std::cout << "\n";
	}
	std::cout << "----------------------" << "\n";
}
double SqrtNorm(const std::vector<double>& vec1, const std::vector<double>& vec2) {
	double sum = 0.;
	for (int i = 0; i < N * N; ++i)
		sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
	return sqrt(sum);
}

int Yacobi_Send_Recive(std::vector<double>& current, std::vector<double>& last, int M, int num_of_first_line, int that_process_id, int count_of_process) {
	int counter = 0;
	double error = 0.;
	do {
		
     	//пересчет
		for (size_t i = 1; i < M - 1; ++i)
			for (size_t j = 1; j < N - 1; ++j)
				current[i * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (i + num_of_first_line) * h ) +
			        last[i * N + j - 1] + last[i * N + j + 1] + last[i * N + j - N] + last[i * N + j + N]);
			
		//считаем ошибку
		double local_err = 0.;
		for (size_t i = N; i < N * M - N; ++i)
			local_err += (current[i] - last[i]) * (current[i] - last[i]);

		MPI_Allreduce(&local_err, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		error = sqrt(error);

		//рассылаем данные
		if (that_process_id != count_of_process - 1) {
			MPI_Send(current.data() + N * (M - 2), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD);
		}
		if (that_process_id != 0) {
			MPI_Recv(current.data(), N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(current.data() + N, N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD);
		}
		if (that_process_id != count_of_process - 1)
			MPI_Recv(current.data() + N * (M - 1), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		++counter;
		current.swap(last);

	} while (error > eps);

	return counter;
}
int Yacobi_SendRecive(std::vector<double>& current, std::vector<double>& last, int M, int num_of_first_line, int that_process_id, int count_of_process) {
	int counter = 0;
	double error = 0.;
	do {

		//пересчет
		for (size_t i = 1; i < M - 1; ++i)
			for (size_t j = 1; j < N - 1; ++j)
				current[i * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (i + num_of_first_line) * h) +
					last[i * N + j - 1] + last[i * N + j + 1] + last[i * N + j - N] + last[i * N + j + N]);

		//считаем ошибку
		double local_err = 0.;
		for (size_t i = N; i < N * M - N; ++i)
			local_err += (current[i] - last[i]) * (current[i] - last[i]);

		MPI_Allreduce(&local_err, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		error = sqrt(error);

		//рассылаем данные
		if (that_process_id != 0)
			MPI_Sendrecv(current.data() + N, N, MPI_DOUBLE, that_process_id - 1, 0, current.data(), N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			
		if (that_process_id != count_of_process - 1)
			MPI_Sendrecv(current.data() + N * (M - 2), N, MPI_DOUBLE, that_process_id + 1, 0, current.data() + N * (M - 1), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		++counter;
		current.swap(last);

	} while (error > eps);

	return counter;
}
int Seidel_Send_Recive(std::vector<double>& current, std::vector<double>& last, int M, int num_of_first_line, int that_process_id, int count_of_process) {
	int counter = 0;
	double error = 0.;
	do {
		//пересчет
		{
			for (size_t i = 1; i < M - 1; ++i)
				for (size_t j = 1 + (num_of_first_line + i + 1) % 2; j < N - 1; j += 2)
					current[i * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (i + num_of_first_line) * h) +
						last[i * N + j - 1] + last[i * N + j + 1] + last[i * N + j - N] + last[i * N + j + N]);
			if (that_process_id != count_of_process - 1) {
				MPI_Send(current.data() + N * (M - 2), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD);
			}
			if (that_process_id != 0) {
				MPI_Recv(current.data(), N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Send(current.data() + N, N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD);
			}
			if (that_process_id != count_of_process - 1)
				MPI_Recv(current.data() + N * (M - 1), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			for (size_t i = 1; i < M - 1; ++i)
				for (size_t j = 1 + (num_of_first_line + i + 2) % 2; j < N - 1; j += 2)
					current[i * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (i + num_of_first_line) * h) +
						current[i * N + j - 1] + current[i * N + j + 1] + current[i * N + j - N] + current[i * N + j + N]);
		}

		//считаем ошибку
		double local_err = 0.;
		for (size_t i = N; i < N * M - N; ++i)
			local_err += (current[i] - last[i]) * (current[i] - last[i]);

		MPI_Allreduce(&local_err, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		error = sqrt(error);

		//рассылаем данные
		if (that_process_id != count_of_process - 1) {
			MPI_Send(current.data() + N * (M - 2), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD);
		}
		if (that_process_id != 0) {
			MPI_Recv(current.data(), N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Send(current.data() + N, N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD);
		}
		if (that_process_id != count_of_process - 1)
			MPI_Recv(current.data() + N * (M - 1), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		++counter;
		current.swap(last);

	} while (error > eps);

	return counter;
}
int Seidel_SendRecive(std::vector<double>& current, std::vector<double>& last, int M, int num_of_first_line, int that_process_id, int count_of_process) {
	int counter = 0;
	double error = 0.;
	do {
		//пересчет
		{
			for (size_t i = 1; i < M - 1; ++i)
				for (size_t j = 1 + (num_of_first_line + i + 1) % 2; j < N - 1; j += 2)
					current[i * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (i + num_of_first_line) * h) +
						last[i * N + j - 1] + last[i * N + j + 1] + last[i * N + j - N] + last[i * N + j + N]);
			if (that_process_id != 0)
				MPI_Sendrecv(current.data() + N, N, MPI_DOUBLE, that_process_id - 1, 0, current.data(), N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			if (that_process_id != count_of_process - 1)
				MPI_Sendrecv(current.data() + N * (M - 2), N, MPI_DOUBLE, that_process_id + 1, 0, current.data() + N * (M - 1), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			for (size_t i = 1; i < M - 1; ++i)
				for (size_t j = 1 + (num_of_first_line + i + 2) % 2; j < N - 1; j += 2)
					current[i * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (i + num_of_first_line) * h) +
						current[i * N + j - 1] + current[i * N + j + 1] + current[i * N + j - N] + current[i * N + j + N]);
		}

		//считаем ошибку
		double local_err = 0.;
		for (size_t i = N; i < N * M - N; ++i)
			local_err += (current[i] - last[i]) * (current[i] - last[i]);

		MPI_Allreduce(&local_err, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		error = sqrt(error);

		//рассылаем данные
		if (that_process_id != 0)
			MPI_Sendrecv(current.data() + N, N, MPI_DOUBLE, that_process_id - 1, 0, current.data(), N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		if (that_process_id != count_of_process - 1)
			MPI_Sendrecv(current.data() + N * (M - 2), N, MPI_DOUBLE, that_process_id + 1, 0, current.data() + N * (M - 1), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		++counter;
		current.swap(last);

	} while (error > eps);

	return counter;
}
int Yacobi_Isend_Irecive(std::vector<double>& current, std::vector<double>& last, int M, int num_of_first_line, int that_process_id, int count_of_process) {

	auto current_first_line = new MPI_Request[2];
	auto last_first_line = new MPI_Request[2];
	if (that_process_id != 0) {
		MPI_Send_init(last.data() + N, N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, last_first_line);
		MPI_Recv_init(last.data(), N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, &last_first_line[1]);

		MPI_Send_init(current.data() + N, N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, current_first_line);
		MPI_Recv_init(current.data(), N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, &current_first_line[1]);
	}
	auto current_last_line = new MPI_Request[2];
	auto last_last_line = new MPI_Request[2];
	if (that_process_id != count_of_process - 1) {
		MPI_Send_init(last.data() + N * (M - 2), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, last_last_line);
		MPI_Recv_init(last.data() + N * (M - 1), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, &last_last_line[1]);

		MPI_Send_init(current.data() + N * (M - 2), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, current_last_line);
		MPI_Recv_init(current.data() + N * (M - 1), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, &current_last_line[1]);
	}
	int counter = 0;
	double error = 0.;
	do {
		if (that_process_id != 0)
			MPI_Startall(2, (counter % 2 == 0) ? last_first_line : current_first_line);
		if (that_process_id != count_of_process - 1)
			MPI_Startall(2, (counter % 2 == 0) ? last_last_line  : current_last_line);



		for (size_t i = 2; i < M - 2; ++i)
			for (size_t j = 1; j < N - 1; ++j)
				current[i * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (i + num_of_first_line) * h) +
					last[i * N + j - 1] + last[i * N + j + 1] + last[i * N + j - N] + last[i * N + j + N]);

		if (that_process_id != 0)
			MPI_Waitall(2, (counter % 2 == 0) ? last_first_line : current_first_line, MPI_STATUSES_IGNORE);
		if (that_process_id != count_of_process - 1)
			MPI_Waitall(2, (counter % 2 == 0) ? last_last_line : current_last_line, MPI_STATUSES_IGNORE);




		for (size_t j = 1; j < N - 1; ++j)
			current[N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (1 + num_of_first_line) * h) +
				last[N + j - 1] + last[N + j + 1] + last[N + j - N] + last[N + j + N]);

		for (size_t j = 1; j < N - 1; ++j)
			current[(M - 2) * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (M - 2 + num_of_first_line) * h) +
				last[(M - 2) * N + j - 1] + last[(M - 2) * N + j + 1] + last[(M - 2) * N + j - N] + last[(M - 2) * N + j + N]);

		double local_err = 0.;
		for (size_t i = N; i < N * M - N; ++i)
			local_err += (current[i] - last[i]) * (current[i] - last[i]);

		MPI_Allreduce(&local_err, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		error = sqrt(error);

		++counter;
		current.swap(last);

	} while (error > eps);
	return counter;
}
int Seidel_Isend_Irecive(std::vector<double>& current, std::vector<double>& last, int M, int num_of_first_line, int that_process_id, int count_of_process) {

	auto current_first_line = new MPI_Request[2];
	auto last_first_line = new MPI_Request[2];
	if (that_process_id != 0) {
		MPI_Send_init(last.data() + N, N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, last_first_line);
		MPI_Recv_init(last.data(), N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, &last_first_line[1]);

		MPI_Send_init(current.data() + N, N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, current_first_line);
		MPI_Recv_init(current.data(), N, MPI_DOUBLE, that_process_id - 1, 0, MPI_COMM_WORLD, &current_first_line[1]);
	}
	auto current_last_line = new MPI_Request[2];
	auto last_last_line = new MPI_Request[2];
	if (that_process_id != count_of_process - 1) {
		MPI_Send_init(last.data() + N * (M - 2), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, last_last_line);
		MPI_Recv_init(last.data() + N * (M - 1), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, &last_last_line[1]);

		MPI_Send_init(current.data() + N * (M - 2), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, current_last_line);
		MPI_Recv_init(current.data() + N * (M - 1), N, MPI_DOUBLE, that_process_id + 1, 0, MPI_COMM_WORLD, &current_last_line[1]);
	}
	int counter = 0;
	double error = 0.;
	do {
		if (that_process_id != 0)
			MPI_Startall(2, (counter % 2 == 0) ? last_first_line : current_first_line);
		if (that_process_id != count_of_process - 1)
			MPI_Startall(2, (counter % 2 == 0) ? last_last_line : current_last_line);
		for (size_t i = 2; i < M - 2; ++i)
			for (size_t j = 1 + (num_of_first_line + i + 1) % 2; j < N - 1; j += 2)
				current[i * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (i + num_of_first_line) * h) +
					last[i * N + j - 1] + last[i * N + j + 1] + last[i * N + j - N] + last[i * N + j + N]);

		if (that_process_id != 0)
			MPI_Waitall(2, (counter % 2 == 0) ? last_first_line : current_first_line, MPI_STATUSES_IGNORE);
		if (that_process_id != count_of_process - 1)
			MPI_Waitall(2, (counter % 2 == 0) ? last_last_line : current_last_line, MPI_STATUSES_IGNORE);

		for (size_t j = 1 + (num_of_first_line + 1 + 1) % 2; j < N - 1; j += 2)
			current[1 * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (1 + num_of_first_line) * h) +
				last[1 * N + j - 1] + last[1 * N + j + 1] + last[1 * N + j - N] + last[1 * N + j + N]);

		for (size_t j = 1 + (num_of_first_line + M - 2 + 1) % 2; j < N - 1; j += 2)
			current[(M - 2) * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, ((M - 2) + num_of_first_line) * h) +
				last[(M - 2) * N + j - 1] + last[(M - 2) * N + j + 1] + last[(M - 2) * N + j - N] + last[(M - 2) * N + j + N]);

		if (that_process_id != 0)
			MPI_Startall(2, (counter % 2 == 1) ? last_first_line : current_first_line);
		if (that_process_id != count_of_process - 1)
			MPI_Startall(2, (counter % 2 == 1) ? last_last_line : current_last_line);

		for (size_t i = 2; i < M - 2; ++i)
			for (size_t j = 1 + (num_of_first_line + i + 2) % 2; j < N - 1; j += 2)
				current[i * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (i + num_of_first_line) * h) +
					current[i * N + j - 1] + current[i * N + j + 1] + current[i * N + j - N] + current[i * N + j + N]);

		if (that_process_id != 0)
			MPI_Waitall(2, (counter % 2 == 1) ? last_first_line : current_first_line, MPI_STATUSES_IGNORE);
		if (that_process_id != count_of_process - 1)
			MPI_Waitall(2, (counter % 2 == 1) ? last_last_line : current_last_line, MPI_STATUSES_IGNORE);

		for (size_t j = 1 + (num_of_first_line + 1 + 2) % 2; j < N - 1; j += 2)
			current[1 * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, (1 + num_of_first_line) * h) +
				current[1 * N + j - 1] + current[1 * N + j + 1] + current[1 * N + j - N] + current[1 * N + j + N]);

		for (size_t j = 1 + (num_of_first_line + (M - 2) + 2) % 2; j < N - 1; j += 2)
			current[(M - 2) * N + j] = (1.0 / (4.0 + k * k * h * h)) * (h * h * f(j * h, ((M - 2) + num_of_first_line) * h) +
				current[(M - 2) * N + j - 1] + current[(M - 2) * N + j + 1] + current[(M - 2) * N + j - N] + current[(M - 2) * N + j + N]);

		double local_err = 0.;
		for (size_t i = N; i < N * M - N; ++i)
			local_err += (current[i] - last[i]) * (current[i] - last[i]);

		MPI_Allreduce(&local_err, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		error = sqrt(error);

		++counter;
		current.swap(last);
	} while (error > eps);
	return counter;
}

template <typename Solver>
std::pair<double, int> Solve(Solver&& solver, const std::string& name_of_method) {
	int that_process_id;
	int count_of_process;

	MPI_Comm_size(MPI_COMM_WORLD, &count_of_process);
	MPI_Comm_rank(MPI_COMM_WORLD, &that_process_id);

	int M = (N - 2) / count_of_process + 2;
	if (that_process_id == count_of_process - 1)
		M = N - (M - 2) * (count_of_process - 1);

	std::vector<double> current(N * M);
	std::vector<double> last(N * M);
    
	int num_of_first_line_for_submatrix = that_process_id * ((N - 2) / count_of_process);
	double t1 = MPI_Wtime();
	int count_of_iteration = solver(current, last, M, num_of_first_line_for_submatrix, that_process_id, count_of_process);
	double time_result = MPI_Wtime() - t1;
	std::vector<double> result(N * N);
	std::vector<int> recvcounts(count_of_process);
	std::vector<int> displs(count_of_process);

	for (int i = 0; i < count_of_process - 1; ++i)
		recvcounts[i] = N * ((N - 2) / count_of_process);

	recvcounts[count_of_process - 1] = N * (N - (M - 2) * (count_of_process - 1) - 2);

	for (size_t i = 0; i < count_of_process; ++i)
		displs[i] = i * N * ((N - 2) / count_of_process);


	MPI_Gatherv(current.data() + N, N * (M - 2), MPI_DOUBLE, result.data() + N, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (that_process_id == 0) {
		std::cout << "Name of method: " << name_of_method << "\n";
		std::cout << "count of processors: " << count_of_process
			<< "; count of iteration: " << count_of_iteration
			<<"; time: "<< time_result
			<< "; abs error: " << SqrtNorm(result, GetExAnswer()) << "\n";
	}
	return { time_result, count_of_iteration };
}


int main(int argc, char* argv[]){


	MPI_Init(&argc, &argv);

	int that_process_id;
	MPI_Comm_rank(MPI_COMM_WORLD, &that_process_id);


	auto res_yacobi_send_resive = Solve(Yacobi_Send_Recive, "Yacobi_Send_Recive");
	auto res_yacobi_sendresive = Solve(Yacobi_SendRecive, "Yacobi_SendRecive");


	MPI_Finalize();
}