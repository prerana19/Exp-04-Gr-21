#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <windows.h>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
# define M_PI 3.1416  /* pi */

using namespace cv;
using namespace std;

// GLOBAL VARIABLES

// FILTER OPTIONS
const int IDEAL_LPF = 0;
const int IDEAL_HPF = 1;
const int GAUSSIAN_LPF = 2;
const int GAUSSIAN_HPF = 3;
const int BUTTER_LPF = 4;
const int BUTTER_HPF = 5;

// FILTER NAMES
const string filters[6] = { "IDEAL_LPF", "IDEAL_HPF", "GAUSSIAN_LPF", "GAUSSIAN_HPF", "BUTTER_LPF", "BUTTER_HPF" };

// GLOBAL PARAMETERS USED BY THE PARAMETERS
int cutoff_G = 1; // 0.1 to 1 @ inc = 0.1
int gaussianSigma_G = 1; // 1 to 100 @ inc = 10
int butterN_G = 1; // 1 to 10 @ inc = 1
int butterC_G = 1; // 0.1 to 1 @ inc = 0.1
int fileID = 0;
int filterID = 0;

vector<string> files;

// MAT DATA STRUCTURES FOR STORING THE IMAGES
Mat image;
Mat operated_img_FFT;
Mat Orig_img_FFT;
Mat FilterImg;
Mat IFFTImg;
float inc = 0.05;

void getfile_list(std::vector<string>& out, const string& directory)
{
	HANDLE dir;
	WIN32_FIND_DATA file_data;
	wchar_t dir_L[256];
	mbstowcs((wchar_t*)dir_L, (directory + "/*.jpg").c_str(), 256);
	if ((dir = FindFirstFile(dir_L, &file_data)) == INVALID_HANDLE_VALUE)
		return; /* No files found */

	do {
		char filename[256];
		wcstombs((char*)filename, file_data.cFileName, 256);
		const string file_name = filename;
		const string full_file_name = directory + "/" + file_name;
		const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

		if (file_name[0] == '.')
			continue;

		if (is_directory)
			continue;

		out.push_back(full_file_name);
	} while (FindNextFile(dir, &file_data));

	FindClose(dir);
}

class comp_float {
public:
	double real;
	double img;

public:
	comp_float()
	{
		this->real = 0;
		this->img = 0;
	}
	comp_float(double real, double img)
	{
		this->real = real;
		this->img = img;
	}
	comp_float operator+(const comp_float& b)
	{
		double r = real + b.real;
		double i = img + b.img;
		return comp_float(r, i);
	}
	comp_float operator-(const comp_float& b)
	{
		double r = real - b.real;
		double i = img - b.img;
		return comp_float(r, i);
	}
	comp_float operator*(const comp_float& b)
	{
		double k1 = b.real * (real + img);
		double k2 = real * (b.img - b.real);
		double k3 = img * (b.img + b.real);
		return comp_float(k1 - k3, k1 + k2);
	}

	comp_float operator*(const double& b)
	{
		return comp_float(real * b, img * b);
	}

	void operator*=(const double& b)
	{
		real *= b;
		img *= b;
	}

	comp_float operator/(const double& b)
	{
		return comp_float(real / b, img / b);
	}

	void operator=(const double& b)
	{
		real = b;
		img = 0;
	}

	double magnitude()
	{
		return sqrt(real * real + img * img);
	}
	void print() {
		cout << real << " + " << img << "i";
	}

};

template<typename T>
void Transpose(T** input_matrix, int N)
{
	T temp;
	for (int i = 0; i < N; i++) {
		T* start = input_matrix[i] + i;
		for (int j = i + 1; j < N; j++) {
			temp = input_matrix[i][j];
			input_matrix[i][j] = input_matrix[j][i];
			input_matrix[j][i] = temp;
		}
	}
}

template<typename T>
void FFTShift(T** input_matrix, int N)
{
	T temp;
	int offset = N / 2;
	for (int i = 0; i < offset; i++) {
		T* start = input_matrix[i] + i;
		for (int j = 0; j < offset; j++) {
			temp = input_matrix[i][j];
			input_matrix[i][j] = input_matrix[i + offset][j + offset];
			input_matrix[i + offset][j + offset] = temp;
		}
	}

	for (int i = N / 2; i < N; i++) {
		T* start = input_matrix[i] + i;
		for (int j = 0; j < offset; j++) {
			temp = input_matrix[i][j];
			input_matrix[i][j] = input_matrix[i - offset][j + offset];
			input_matrix[i - offset][j + offset] = temp;
		}
	}
}

template<typename T>
void FFTShift(Mat& input_matrix, int N)
{
	T temp;
	int offset = N / 2;
	for (int i = 0; i < offset; i++) {
		for (int j = 0; j < offset; j++) {
			temp = input_matrix.at<T>(i, j);
			input_matrix.at<T>(i, j) = input_matrix.at<T>(i + offset, j + offset);
			input_matrix.at<T>(i + offset, j + offset) = temp;
		}
	}

	for (int i = N / 2; i < N; i++) {
		for (int j = 0; j < offset; j++) {
			temp = input_matrix.at<T>(i, j);
			input_matrix.at<T>(i, j) = input_matrix.at<T>(i - offset, j + offset);
			input_matrix.at<T>(i - offset, j + offset) = temp;
		}
	}
}

comp_float* FFT(uchar* x, int N, int arrSize, int zeroLoc, int gap)
{
	comp_float* fft;
	fft = new comp_float[N];

	int i;
	if (N == 2)
	{
		fft[0] = comp_float(x[zeroLoc] + x[zeroLoc + gap], 0);
		fft[1] = comp_float(x[zeroLoc] - x[zeroLoc + gap], 0);
	}
	else
	{
		comp_float wN = comp_float(cos(2 * M_PI / N), sin(-2 * M_PI / N));//exp(-j2*pi/N)
		comp_float w = comp_float(1, 0);
		gap *= 2;
		comp_float* X_even = FFT(x, N / 2, arrSize, zeroLoc, gap); //N/2 POINT DFT OF EVEN X's
		comp_float* X_odd = FFT(x, N / 2, arrSize, zeroLoc + (arrSize / N), gap); //N/2 POINT DFT OF ODD X's
		comp_float todd;
		for (i = 0; i < N / 2; ++i)
		{
			todd = w * X_odd[i];
			fft[i] = X_even[i] + todd;
			fft[i + N / 2] = X_even[i] - todd;
			w = w * wN;
		}

		delete[] X_even;
		delete[] X_odd;
	}

	return fft;
}

comp_float* FFT(comp_float* x, int N, int arrSize, int zeroLoc, int gap)
{
	comp_float* fft;
	fft = new comp_float[N];

	int i;
	if (N == 2)
	{
		fft[0] = x[zeroLoc] + x[zeroLoc + gap];
		fft[1] = x[zeroLoc] - x[zeroLoc + gap];
	}
	else
	{
		comp_float wN = comp_float(cos(2 * M_PI / N), sin(-2 * M_PI / N));//exp(-j2*pi/N)
		comp_float w = comp_float(1, 0);
		gap *= 2;
		comp_float* X_even = FFT(x, N / 2, arrSize, zeroLoc, gap); //N/2 POINT DFT OF EVEN X's
		comp_float* X_odd = FFT(x, N / 2, arrSize, zeroLoc + (arrSize / N), gap); //N/2 POINT DFT OF ODD X's
		comp_float todd;
		for (i = 0; i < N / 2; ++i)
		{
			todd = w * X_odd[i];
			fft[i] = X_even[i] + todd;
			fft[i + N / 2] = X_even[i] - todd;
			w = w * wN;
		}

		delete[] X_even;
		delete[] X_odd;
	}

	return fft;
}

comp_float* IFFT(comp_float* fft, int N, int arrSize, int zeroLoc, int gap)
{
	comp_float* signal;
	signal = new comp_float[N];

	int i;
	if (N == 2)
	{
		signal[0] = fft[zeroLoc] + fft[zeroLoc + gap];
		signal[1] = fft[zeroLoc] - fft[zeroLoc + gap];
	}
	else
	{
		comp_float wN = comp_float(cos(2 * M_PI / N), sin(2 * M_PI / N));//exp(j2*pi/N)
		comp_float w = comp_float(1, 0);
		gap *= 2;
		comp_float* X_even = IFFT(fft, N / 2, arrSize, zeroLoc, gap); //N/2 POINT DFT OF EVEN X's
		comp_float* X_odd = IFFT(fft, N / 2, arrSize, zeroLoc + (arrSize / N), gap); //N/2 POINT DFT OF ODD X's
		comp_float todd;
		for (i = 0; i < N / 2; ++i)
		{
			todd = w * X_odd[i];
			signal[i] = (X_even[i] + todd) * 0.5;
			signal[i + N / 2] = (X_even[i] - todd) * 0.5;
			w = w * wN; // Get the next root(conjugate) among Nth roots of unity
		}

		delete[] X_even;
		delete[] X_odd;
	}

	return signal;
}

comp_float** FFT2(Mat& orig_image) {
	cout << "Applying FFT2" << endl;

	if (orig_image.rows != orig_image.cols) {
		cout << "Image is not Valid";
		return nullptr;
	}
	int N = orig_image.rows;
	//cout << "Image size:" << N << endl;
	comp_float** FFT2Result_h;
	FFT2Result_h = new comp_float * [N];

	// ROW WISE FFT
	for (int i = 0; i < N; i++) {
		uchar* row = orig_image.ptr<uchar>(i);
		FFT2Result_h[i] = FFT(row, N, N, 0, 1);
	}

	//cout << "final: " << endl;
	Transpose<comp_float>(FFT2Result_h, N);

	// COLUMN WISE FFT
	for (int i = 0; i < N; i++) {
		FFT2Result_h[i] = FFT(FFT2Result_h[i], N, N, 0, 1);
	}
	Transpose<comp_float>(FFT2Result_h, N);

	return FFT2Result_h;
}

comp_float** IFFT2(comp_float** orig_image, int N) {

	cout << "Applying IFFT2" << endl;

	comp_float** ifftResult;
	ifftResult = new comp_float * [N];
	// ROW WISE FFT
	for (int i = 0; i < N; i++) {
		ifftResult[i] = IFFT(orig_image[i], N, N, 0, 1);
	}

	//cout << "final: " << endl;
	Transpose<comp_float>(ifftResult, N);

	int d = N * N;
	// COLUMN WISE FFT
	for (int i = 0; i < N; i++) {
		ifftResult[i] = IFFT(ifftResult[i], N, N, 0, 1);
		for (int j = 0; j < N; j++) {
			ifftResult[i][j] = ifftResult[i][j] / d;
		}
	}
	Transpose<comp_float>(ifftResult, N);

	cout << endl;

	return ifftResult;
}

void comp_to_mat(comp_float** orig_image, Mat& dest, int N, bool shift = false, float maxF = 1.0) {
	if (shift) {
		FFTShift(orig_image, N);
	}
	dest = Mat(N, N, CV_32F, cv::Scalar::all(0));
	float min = 99999;
	float max = 0;

	// Find min and max
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			orig_image[i][j] = orig_image[i][j] / N;
			float m = orig_image[i][j].magnitude();
			if (m < min) {
				min = m;
			}
			if (m > max) {
				max = m;
			}
		}
	}


	// Normalize the image
	float range = (max - min);
	for (int i = 0; i < N; i++) {
		float* p = dest.ptr<float>(i);
		for (int j = 0; j < N; j++) {
			p[j] = (orig_image[i][j].magnitude() - min) * maxF / range;
		}
	}
	//cout << "Min: " << min << " Max:" << max;
}

void ApplyFilter(comp_float** orig_image, Mat& filterImg, int N, int FilterType) {
	float cutoff = cutoff_G * inc; //Compute Ideal filter cutoff
	float sigma_squared = gaussianSigma_G * inc + inc; //Compute Gaussing filter sigma from trackbar input
	int butter_n = butterN_G; // Butterworth parameter n
	// Cutoff lies in [0, 2]
	cutoff *= cutoff; // Square it to avoid further sqrt
	filterImg = Mat(N, N, CV_32F); // Image for showing the frequency spectrum of the filter
	float d = N * N;
	comp_float** filterFFT;
	switch (FilterType) {
	case IDEAL_LPF:
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float f = (i * i / d) + (j * j / d);
				if (i == 5 && j == 5) {
					cout << "Cutoff Frequency is:" << f << endl;
				}
				if (f > cutoff) {
					// Remove the components outside the
					// cutoff frequency
					orig_image[i][j] = 0;
					orig_image[N - 1 - i][N - 1 - j] = 0;
					orig_image[N - 1 - i][j] = 0;
					orig_image[i][N - 1 - j] = 0;
				}
				else {
					// Filter coeff = 1 withing cutoff frequency range
					filterImg.at<float>(i, j) = filterImg.at<float>(N - 1 - i, N - 1 - j) = filterImg.at<float>(N - 1 - i, j) = filterImg.at<float>(i, N - 1 - j) = 1;
				}
			}
		}
		break;
	case IDEAL_HPF:
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float f = (i * i / d) + (j * j / d);
				if (i == 5 && j == 5) {
					cout << "Cutoff Frequency is:" << f << endl;
				}
				if (f <= cutoff) {
					// Remove the components @ less than the
					// cutoff frequency
					orig_image[i][j] = 0;
					orig_image[N - 1 - i][N - 1 - j] = 0;
					orig_image[N - 1 - i][j] = 0;
					orig_image[i][N - 1 - j] = 0;
				}
				else {
					// Filter coeff = 1 withing cutoff frequency range
					filterImg.at<float>(i, j) = filterImg.at<float>(N - 1 - i, N - 1 - j) = filterImg.at<float>(N - 1 - i, j) = filterImg.at<float>(i, N - 1 - j) = 1;
				}
			}
		}
		break;
	case GAUSSIAN_LPF:
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float wx2 = pow(2 * M_PI * i / N, 2); // omega x squared
				float wy2 = pow(2 * M_PI * j / N, 2); // omega y squared
				float coeff = exp(-(wx2 + wy2) / (2 * sigma_squared)); // Gaussian filter coeff @ (wx, wy)
				orig_image[i][j] *= coeff;
				orig_image[N - 1 - i][N - 1 - j] *= coeff;
				orig_image[N - 1 - i][j] *= coeff;
				orig_image[i][N - 1 - j] *= coeff;

				filterImg.at<float>(i, j) = filterImg.at<float>(N - 1 - i, N - 1 - j) = filterImg.at<float>(N - 1 - i, j) = filterImg.at<float>(i, N - 1 - j) = coeff;
			}
		}
		break;
	case GAUSSIAN_HPF:
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float wx2 = pow(2 * M_PI * i / N, 2); // omega x squared
				float wy2 = pow(2 * M_PI * j / N, 2); // omega y squared
				float coeff = 1 - exp(-(wx2 + wy2) / (2 * sigma_squared)); // Gaussian filter coeff @ (wx, wy)
				orig_image[i][j] *= coeff;
				orig_image[N - 1 - i][N - 1 - j] *= coeff;
				orig_image[N - 1 - i][j] *= coeff;
				orig_image[i][N - 1 - j] *= coeff;

				filterImg.at<float>(i, j) = filterImg.at<float>(N - 1 - i, N - 1 - j) = filterImg.at<float>(N - 1 - i, j) = filterImg.at<float>(i, N - 1 - j) = coeff;

			}
		}
		break;
	case BUTTER_LPF:
		cutoff = pow((butterC_G * inc + inc) * M_PI, 2);
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float wx2 = pow(2 * M_PI * i / N, 2);
				float wy2 = pow(2 * M_PI * j / N, 2);
				float coeff = 1 / (1 + pow((wx2 + wy2) / cutoff, 2 * butter_n)); // Butterworth filter coeff @ (wx, wy)
				orig_image[i][j] *= coeff;
				orig_image[N - 1 - i][N - 1 - j] *= coeff;
				orig_image[N - 1 - i][j] *= coeff;
				orig_image[i][N - 1 - j] *= coeff;

				filterImg.at<float>(i, j) = filterImg.at<float>(N - 1 - i, N - 1 - j) = filterImg.at<float>(N - 1 - i, j) = filterImg.at<float>(i, N - 1 - j) = coeff;

			}
		}
		break;
	case BUTTER_HPF:
		cutoff = pow((butterC_G * inc + inc) * M_PI, 2);
		for (int i = 0; i < N / 2; i++) {
			for (int j = 0; j < N / 2; j++) {
				float wx2 = pow(2 * M_PI * i / N, 2);
				float wy2 = pow(2 * M_PI * j / N, 2);
				float coeff = 1 / (1 + pow(cutoff / (wx2 + wy2), 2 * butter_n)); // Butterworth filter coeff @ (wx, wy)
				orig_image[i][j] *= coeff;
				orig_image[N - 1 - i][N - 1 - j] *= coeff;
				orig_image[N - 1 - i][j] *= coeff;
				orig_image[i][N - 1 - j] *= coeff;

				filterImg.at<float>(i, j) = filterImg.at<float>(N - 1 - i, N - 1 - j) = filterImg.at<float>(N - 1 - i, j) = filterImg.at<float>(i, N - 1 - j) = coeff;

			}
		}
		break;
	}
}

// Event handlers for trackbars
void apply_filter(string filename) {

	if (!image.data)
	{
		cout << "Could not open or find the image" << std::endl;
		return;
	}
	switch (filterID) {
	case IDEAL_LPF:
		cout << "Applying IDEAL_LPF" << endl;
		break;
	case IDEAL_HPF:
		cout << "Applying IDEAL_HPF" << endl;
		break;
	case GAUSSIAN_LPF:
		cout << "Applying GAUSSIAN_LPF" << endl;
		break;
	case GAUSSIAN_HPF:
		cout << "Applying GAUSSIAN_HPF" << endl;
		break;
	case BUTTER_LPF:
		cout << "Applying BUTTER_LPF" << endl;
		break;
	case BUTTER_HPF:
		cout << "Applying BUTTER_HPF" << endl;
		break;
	}
	comp_float** fft2result = FFT2(image);
	comp_to_mat(fft2result, Orig_img_FFT, image.rows, false, 255);
	// Apply Filter
	ApplyFilter(fft2result, FilterImg, image.rows, filterID);
	// Roate the FFT input_matrix to bring freq(0, 0) at the middle of the image
	FFTShift<float>(FilterImg, image.rows);
	//ApplyFilter(fft2result, image.rows, IDEAL_LPF, 0.15);

	comp_float** ifft2result = IFFT2(fft2result, image.rows);
	float maxF = 1;
	switch (filterID) {
	case IDEAL_LPF:
	case GAUSSIAN_LPF:
	case BUTTER_LPF:
		maxF = 255;
	}
	comp_to_mat(fft2result, operated_img_FFT, image.rows, true, maxF);
	comp_to_mat(ifft2result, IFFTImg, image.rows);


	FFTShift<float>(Orig_img_FFT, image.rows);

	// Show the results
	imshow("FFT window Before", Orig_img_FFT);
	imshow("FFT window After", operated_img_FFT);
	cout << "Showing Filter Spectrum" << endl;
	imshow("Filter Spectrum", FilterImg);
	cout << "Showing Output image" << endl;
	imshow("Output Image", IFFTImg);
}

void on_cutoff_change(int, void*) {
	cout << "Cutoff frequency chosen: " << cutoff_G * inc + inc << endl;
	apply_filter(files[fileID]);
}

void on_gSigma_change(int, void*) {
	cout << "Sigma Value: " << gaussianSigma_G * inc + inc << endl;
	apply_filter(files[fileID]);
}

void on_butterN_change(int, void*) {
	cout << "ButterWorth Order: " << butterN_G << endl;
	apply_filter(files[fileID]);
}

void on_butterC_change(int, void*) {
	cout << "ButterWorth Order: " << (butterC_G * inc + inc) << endl;
	apply_filter(files[fileID]);
}
void on_file_change(int, void*) {
	cout << "File: " << files[fileID] << endl;

	image = imread(files[fileID], IMREAD_GRAYSCALE); // Read the file

	if (!image.data) // Check for invalid input
	{
		cout << "Could not open or find the image" << endl;
		return;
	}
	imshow("orig_image File", image); // Show our image inside it.
	apply_filter(files[fileID]);
}

void on_filter_hange(int, void*) {
	cout << "Filter: " << filters[filterID] << endl;
	apply_filter(files[fileID]);
}

int main()
{
	int displaySize = 400;
	string folder;

	folder = "./TestImages";
	getfile_list(files, folder);

	namedWindow("orig_image File", 0);
	namedWindow("parameters", 0);
	//resizeWindow("orig_image File", displaySize, displaySize);
	// Create trackbars
	createTrackbar("Image", "parameters", &fileID, files.size() - 1, on_file_change);
	createTrackbar("FilterID", "parameters", &filterID, 5, on_filter_hange);
	createTrackbar("Ideal Filter cutoff ", "parameters", &cutoff_G, 0.5 / inc, on_cutoff_change); // cutoff
	createTrackbar("Gaussian Filter Sigma ", "parameters", &gaussianSigma_G, 2 / inc, on_gSigma_change); // gaussian sigma
	createTrackbar("ButterWorth n ", "parameters", &butterN_G, 10, on_butterN_change); // butterworth n
	createTrackbar("ButterWorth c", "parameters", &butterC_G, 1 / inc, on_butterC_change); // butterworth c

	// Create the windows for showing results
	namedWindow("Filter Spectrum", 0);
	//resizeWindow("Filter Spectrum", displaySize, displaySize);

	namedWindow("Output Image", 0);
	//resizeWindow("Output Image", displaySize, displaySize);

	waitKey(0);
	return 0;
}