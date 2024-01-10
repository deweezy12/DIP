//============================================================================
// Name        : Dip5.cpp
// Author      : Ronny Haensch, Andreas Ley
// Version     : 3.0
// Copyright   : -
// Description : 
//============================================================================

#include "Dip5.h"


namespace dip5 {
    

/**
* @brief Generates gaussian filter kernel of given size
* @param kSize Kernel size (used to calculate standard deviation)
* @returns The generated filter kernel
*/
cv::Mat_<float> createGaussianKernel1D(float sigma)
{
    unsigned kSize = getOddKernelSizeForSigma(sigma);
    // Hopefully already DONE, copy from last homework, just make sure you compute the kernel size from the given sigma (and not the other way around)
    // TO DO !!!
    cv::Mat_<float> vector = cv::Mat_<float>::zeros(1, kSize);
    float pi = CV_PI;
    float normalizer = 0;
    for (int i = 0; i < kSize; i++) {
        vector.at<float>(i) = 1 / (2.0 * pi * sigma) * exp(-1 / 2.0 * (i - kSize / (int)2) * (i - kSize / (int)2) / sigma / sigma);
        normalizer += vector.at<float>(i);
    }
    vector = 1 / normalizer * vector;
    //std::cout<< "vector = " << vector <<std::endl;   
    return vector;
    
}
cv::Mat_<float> spatialConvolution(const cv::Mat_<float>& src, const cv::Mat_<float>& kernel)
{

       // TO DO !!
    float normal_sum = 0;
    for (int a = 0; a < kernel.rows; a++) {
        for (int b = 0; b < kernel.cols; b++) {
            normal_sum += kernel.at<float>(a, b);
        }
    }
    // Zero padding

    cv::Mat output_image = cv::Mat::zeros(src.rows, src.cols, CV_32FC1);
    int kernelRows = kernel.rows;
    int kernelCols = kernel.cols;


    int startRows = kernelRows / 2;
    int startCols = kernelCols / 2;

    cv::Mat zero_padded;

    cv::copyMakeBorder(src, zero_padded, startRows, startRows, startCols, startCols, cv::BORDER_CONSTANT, cv::Scalar(0));

    // Flip the kernel
    cv::Mat flipedKernel;
    cv::flip(kernel, flipedKernel, -1); // -1 to flip it both horizontally and vertically
    for (int i = startRows; i < zero_padded.rows - startRows; i++) {
        for (int j = startCols; j < zero_padded.cols - startCols; j++) {
            float summe = 0;
            for (int x = 0; x < kernelRows; x++) {
                for (int y = 0; y < kernelCols; y++) {
                    summe += zero_padded.at<float>(i + x - startRows, j + y - startCols) * flipedKernel.at<float>(x, y);
                }
            }
            if (normal_sum == 1) {    // the value 1 should be changed according to the filter
                output_image.at<float>(i - startRows, j - startCols) = summe;
            }
            else {
                output_image.at<float>(i - startRows, j - startCols) = summe / normal_sum;
            }
        }
    }

    return output_image;
}

/**
* @brief Convolution in spatial domain by seperable filters
* @param src Input image
* @param size Size of filter kernel
* @returns Convolution result
*/
cv::Mat_<float> separableFilter(const cv::Mat_<float>& src, const cv::Mat_<float>& kernelX, const cv::Mat_<float>& kernelY)
{
    // Hopefully already DONE, copy from last homework
    // But do mind that this one gets two different kernels for horizontal and vertical convolutions.
    cv::Mat_<float> matX = spatialConvolution(src, kernelX);
    cv::transpose(matX, matX);
    cv::Mat_<float> matY = spatialConvolution(matX, kernelY);
    cv::transpose(matY, matY);

    return matY;
    
}

/**
 * @brief Creates kernel representing fst derivative of a Gaussian kernel (1-dimensional)
 * @param sigma standard deviation of the Gaussian kernel
 * @returns the calculated kernel
 */
cv::Mat_<float> createFstDevKernel1D(float sigma) 
{
    unsigned kSize = getOddKernelSizeForSigma(sigma);
    // TO DO !!!
    cv::Mat_<float> kernel = cv::Mat_<float>::zeros(1, kSize);
    const double pi = 3.141592653589793238462643383279502884197;
    // The first part of the first derivative as the other part is the gauss formula
    double firstElement;

    // Positive value should be on one side and the negative on the other.
    int center = (kSize - 1) / 2;
    // Normalized the kernel so the sum of all element is 1
    double sum = 0.0;

    for (int i = 0; i < kSize; i++) {
        int x = i - center;
        double gauss = (1.0 / (sqrt(2 * pi) * sigma) * exp(-(x * x) / (2 * sigma * sigma)));
        firstElement = -x / (sigma * sigma);
        kernel.at<float>(0, i) = firstElement * gauss;
        sum += kernel.at<float>(0, i);
    }

    // Normalize the kernel;
    for (int j = 0; j < kSize; j++) {
        kernel.at<float>(0, j) /= sum;
    }

    return kernel;
}


/**
 * @brief Calculates the directional gradients through convolution
 * @param img The input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param gradX Matrix through which to return the x component of the directional gradients
 * @param gradY Matrix through which to return the y component of the directional gradients
 */
void calculateDirectionalGradients(const cv::Mat_<float>& img, float sigmaGrad,
                            cv::Mat_<float>& gradX, cv::Mat_<float>& gradY)
{
    // TO DO !!!

    gradX.create(img.rows, img.cols);
    gradY.create(img.rows, img.cols);

    cv::Mat gaussianKernel = createGaussianKernel1D(sigmaGrad);
    cv::Mat gaussianKernelDerivative = createFstDevKernel1D(sigmaGrad);

    // for x convolve in h with GD and V with NG
    gradX = separableFilter(img, gaussianKernelDerivative, gaussianKernel);
    
    // for y convolve in h with NG and V with GD
    gradY = separableFilter(img, gaussianKernel, gaussianKernelDerivative);

}

/**
 * @brief Calculates the structure tensors (per pixel)
 * @param gradX The x component of the directional gradients
 * @param gradY The y component of the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation".
 * @param A00 Matrix through which to return the A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 Matrix through which to return the A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 Matrix through which to return the A_{1,1} elements of the structure tensor of each pixel.
 */
void calculateStructureTensor(const cv::Mat_<float>& gradX, const cv::Mat_<float>& gradY, float sigmaNeighborhood,
                            cv::Mat_<float>& A00, cv::Mat_<float>& A01, cv::Mat_<float>& A11)
{
    A00.create(gradX.rows, gradX.cols);
    A01.create(gradX.rows, gradX.cols);
    A11.create(gradX.rows, gradX.cols);

    // TO DO !!!
    for (int i = 0; i < gradX.rows; i++) {
        for (int j = 0; j < gradX.cols; j++) {
            A00.at<float>(i, j) = gradX.at<float>(i, j) *  gradX.at<float>(i, j);
            A11.at<float>(i, j) = gradY.at<float>(i, j) *  gradY.at<float>(i, j);
            A01.at<float>(i, j) = gradX.at<float>(i, j) *  gradX.at<float>(i, j);
        }
    }
    cv::Mat_<float> gaussianKernel = createGaussianKernel1D(sigmaNeighborhood);
    A00 = separableFilter(A00, gaussianKernel, gaussianKernel);
    A01 = separableFilter(A01, gaussianKernel, gaussianKernel);
    A11 = separableFilter(A11, gaussianKernel, gaussianKernel);

}

/**
 * @brief Calculates the feature point weight and isotropy from the structure tensors.
 * @param A00 The A_{0,0} elements of the structure tensor of each pixel.
 * @param A01 The A_{0,1} elements of the structure tensor of each pixel.
 * @param A11 The A_{1,1} elements of the structure tensor of each pixel.
 * @param weight Matrix through which to return the weights of each pixel.
 * @param isotropy Matrix through which to return the isotropy of each pixel.
 */
void calculateFoerstnerWeightIsotropy(const cv::Mat_<float>& A00, const cv::Mat_<float>& A01, const cv::Mat_<float>& A11,
                                    cv::Mat_<float>& weight, cv::Mat_<float>& isotropy)
{
    weight.create(A00.rows, A00.cols);
    isotropy.create(A00.rows, A00.cols);

    for (int i = 0; i < A00.rows; i++) {
        for (int j = 0; j < A00.cols; j++) {
            // Calculate determinant and trace of the structure tensor
            float detA = A00.at<float>(i, j) * A11.at<float>(i, j) - A01.at<float>(i, j) * A01.at<float>(i, j);
            float traceA = A00.at<float>(i, j) + A11.at<float>(i, j);

            // Calculate the weight and isotropy
            weight.at<float>(i, j) = detA / traceA;
            isotropy.at<float>(i, j) = traceA > 0 ? (traceA * traceA) / detA : 0;
        }
    }
}



/**
 * @brief Finds Foerstner interest points in an image and returns their location.
 * @param img The greyscale input image
 * @param sigmaGrad The standard deviation of the Gaussian kernel for the directional gradients
 * @param sigmaNeighborhood The standard deviation of the Gaussian kernel for computing the "neighborhood summation" of the structure tensor.
 * @param fractionalMinWeight Threshold on the weight as a fraction of the mean of all locally maximal weights.
 * @param minIsotropy Threshold on the isotropy of interest points.
 * @returns List of interest point locations.
 */
std::vector<cv::Vec2i> getFoerstnerInterestPoints(const cv::Mat_<float>& img, float sigmaGrad, float sigmaNeighborhood, float fractionalMinWeight, float minIsotropy)
{
    std::vector<cv::Vec2i> interestPoints;

    // Calculate directional gradients
    cv::Mat_<float> gradX, gradY;
    dip5::calculateDirectionalGradients(img, sigmaGrad, gradX, gradY);

    // Calculate structure tensors
    cv::Mat_<float> A00, A01, A11;
    dip5::calculateStructureTensor(gradX, gradY, sigmaNeighborhood, A00, A01, A11);

    // Calculate Foerstner weight and isotropy
    cv::Mat_<float> weight, isotropy;
    dip5::calculateFoerstnerWeightIsotropy(A00, A01, A11, weight, isotropy);

    // Compute threshold as a fraction of the mean of all locally maximal weights
    float threshold = fractionalMinWeight * cv::mean(weight)[0];

    // Find interest points based on thresholds
    for (int i = 0; i < weight.rows; i++) {
        for (int j = 0; j < weight.cols; j++) {
            if (weight(i, j) > threshold && isotropy(i, j) > minIsotropy && dip5::isLocalMaximum(weight, j, i)) {
                interestPoints.push_back(cv::Vec2i(j, i));
            }
        }
    }

    return interestPoints;
}



/* *****************************
  GIVEN FUNCTIONS
***************************** */


// Use this to compute kernel sizes so that the unit tests can simply hard checks for correctness.
unsigned getOddKernelSizeForSigma(float sigma)
{
    unsigned kSize = (unsigned) std::ceil(5.0f * sigma) | 1;
    if (kSize < 3) kSize = 3;
    return kSize;
}

bool isLocalMaximum(const cv::Mat_<float>& weight, int x, int y)
{
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++) {
            int x_ = std::min(std::max(x+j, 0), weight.cols-1);
            int y_ = std::min(std::max(y+i, 0), weight.rows-1);
            if (weight(y_, x_) > weight(y, x))
                return false;
        }
    return true;
}

}
