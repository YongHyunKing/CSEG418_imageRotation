
#include <iostream>
#include <opencv2\opencv.hpp>
#include <cmath>

/* Find Boundary of output image */
cv::Point2d output_Boundary(cv::Mat img,cv::Mat rotation_matrix )
{
    /* get the angular points of original image */
	cv::Point2d arr[4] = { cv::Point2d(0, 0),
		cv::Point2d(img.cols, 0),
		cv::Point2d(0, img.rows),
		cv::Point2d(img.cols, img.rows) };
    
	double min_x = img.cols, max_x = 0;
	double min_y = img.rows, max_y = 0;
	cv::Point2d center (img.cols / 2.0, img.rows / 2.0);

	for (int i = 0; i < 4; i++)
	{
		double x(center.x + (arr[i].x - center.x) * rotation_matrix.at<double>(0, 0) + (arr[i].y - center.y) * rotation_matrix.at<double>(0, 1));
		double y(center.y + (arr[i].x - center.x) * rotation_matrix.at<double>(1, 0) + (arr[i].y - center.y) * rotation_matrix.at<double>(1, 1));
		min_x = cv::min(min_x, x);
		max_x = cv::max(max_x, x);
		min_y = cv::min(min_y, y);
		max_y = cv::max(max_y, y);
	}

	return cv::Point2d(max_x - min_x, max_y - min_y);
}

/* Image rotation using forward method */
cv::Mat problem_a_rotate_forward(cv::Mat img, double angle) {

	cv::Mat output(img.size(),img.type(), cv::Scalar(0));
	double degree = angle * CV_PI / 180.;

    /* make rotation matrix (type : double) */
	double rotation[] = { cos(degree),-sin(degree), sin(degree),cos(degree) };
	cv::Mat rotation_matrix(2, 2, CV_64F, rotation);
    
    /* get range of output image */
	cv::Point2d range = output_Boundary(img, rotation_matrix);
    
    /* make padding */
	cv::Point2d padding(range.x / 2.0 - img.cols / 2.0, range.y / 2.0 - img.rows / 2.0);
	
    /* resize output image */
	resize(output, output, cv::Size(cv::abs(range.x), cv::abs(range.y)));
    
    /* get center of original image */
    cv::Point2d center(img.cols / 2.0, img.rows / 2.0);
	
	/* input original image data to the rotated point of output image */
	for (int old_x = 0; old_x < img.cols; old_x++) {
		for (int old_y = 0; old_y < img.rows; old_y++) {
			double new_x = (center.x + (old_x - center.x) * rotation_matrix.at<double>(0, 0) + (old_y - center.y) * rotation_matrix.at<double>(0, 1)) + padding.x;
			double new_y = (center.y + (old_x - center.x) * rotation_matrix.at<double>(1, 0) + (old_y - center.y) * rotation_matrix.at<double>(1, 1)) + padding.y;

			if ((new_x < 0.0) || (new_x >= output.cols) || (new_y < 0.0) || (new_y >= output.rows)) continue;
			
			output.at<cv::Vec3b>(new_y, new_x) = img.at<cv::Vec3b>(old_y, old_x);

		}
	}

	cv::imshow("a_output", output); cv::waitKey(0);
	return output;
}
	
/* Image rotation using backward method */
cv::Mat problem_b_rotate_backward(cv::Mat img, double angle){
	cv::Mat output(img.size(), img.type(), cv::Scalar(0));
	double degree = angle * CV_PI / 180.;

    /* make rotation matrix (type : double) */
	double rotation[] = { cos(degree),-sin(degree), sin(degree),cos(degree) };
	cv::Mat rotation_matrix(2, 2, CV_64F, rotation);
    
    /* get reverse rotation matrix */
	cv::Mat reverse = rotation_matrix.inv();
    
    /* get range of output image */
	cv::Point2d range = output_Boundary(img, reverse);

    /* resize output image */
	resize(output, output, cv::Size(cv::abs(range.x), cv::abs(range.y)));

    /* get center of output image */
	cv::Point2d center(output.cols / 2.0, output.rows / 2.0);
    
    /* make padding */
	cv::Point2d padding(range.x / 2.0 - img.cols / 2.0, range.y / 2.0 - img.rows / 2.0 );

    /* output image gets the data from the point of original image which rotated reversely*/
	for (int new_x = 0; new_x < output.cols; new_x++) {
		for (int new_y = 0; new_y < output.rows; new_y++) {
			double old_x = (center.x + (new_x - center.x) * reverse.at<double>(0, 0) + (new_y - center.y) * reverse.at<double>(0, 1)) - padding.x;
			double old_y = (center.y + (new_x - center.x) * reverse.at<double>(1, 0) + (new_y - center.y) * reverse.at<double>(1, 1)) - padding.y;

			if (old_x < 0 || old_y < 0 || old_x >= img.cols || old_y >= img.rows) continue;
			if (new_x < 0 || new_y < 0 || new_x >= output.cols || new_y >= output.rows) continue;

			output.at<cv::Vec3b>(new_y, new_x) = img.at<cv::Vec3b>(old_y, old_x);

		}
	}
	cv::imshow("b_output", output); cv::waitKey(0);
	return output;
}

/* Image rotation using backward method and interpolation */
cv::Mat problem_c_rotate_backward_interarea(cv::Mat img, double angle){
    cv::Mat output(img.size(), img.type(), cv::Scalar(0));
    double degree = angle * CV_PI / 180.;

    /* make rotation matrix (type : double) */
    double rotation[] = { cos(degree),-sin(degree), sin(degree),cos(degree) };
    cv::Mat rotation_matrix(2, 2, CV_64F, rotation);
    
    /* get reverse rotation matrix */
    cv::Mat reverse = rotation_matrix.inv();
    
    /* get range of output image */
    cv::Point2d range = output_Boundary(img, reverse);

    /* resize output image */
    resize(output, output, cv::Size(cv::abs(range.x), cv::abs(range.y)));

    /* get center of output image */
    cv::Point2d center(output.cols / 2.0, output.rows / 2.0);
    
    /* make padding */
    cv::Point2d padding(range.x / 2.0 - img.cols / 2.0, range.y / 2.0 - img.rows / 2.0 );

    /* output image gets the data from the point of original image which rotated reversely*/
	for (int new_x = 0; new_x < output.cols; new_x++) {
		for (int new_y = 0; new_y < output.rows; new_y++) {
			double old_x = (center.x + (new_x - center.x) * reverse.at<double>(0, 0) + (new_y - center.y) * reverse.at<double>(0, 1)) - padding.x;
			double old_y = (center.y + (new_x - center.x) * reverse.at<double>(1, 0) + (new_y - center.y) * reverse.at<double>(1, 1)) - padding.y;
            
            /* get the angular points */
			double border[2][2] = { {floor(old_x),ceil(old_x)},{floor(old_y),ceil(old_y)} };


			if (border[0][0] < 0 || border[0][0] >= img.cols || border[0][1] < 0 || border[0][1] >= img.cols) continue;
			if (border[1][0] < 0 || border[1][0] >= img.rows || border[1][1] < 0 || border[1][1] >= img.rows) continue;

			cv::Vec3b border_image[4];
            
            /* save the date of origin image */
			int index = 0;
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					border_image[index++] = img.at<cv::Vec3b>(border[1][i], border[0][j]);
				}
			}

			double diff_x(old_x - border[0][0]);
			double diff_y(old_y - border[1][0]);

            /* save the data which is weighted by interpolation method to output image */
            for (int i = 0; i < 3; i++) {/* R = 0, G = 1, B = 2 */
				int value = (int)ceil(((1 - diff_y) * ((1 - diff_x) * border_image[0][i] + diff_x * border_image[1][i]))
					+ (diff_y * ((1 - diff_x) * border_image[2][i] + diff_x * border_image[3][i])));
				output.at<cv::Vec3b>(new_y, new_x)[i] = value > 255 ? 255 : value < 0 ? 0 : value;
			}



		}
	}

	cv::imshow("c_output", output); cv::waitKey(0);

	return output;
}

int main(void){

	double angle = -15.0f;

	cv::Mat input = cv::imread("lena.jpg");
	//Fill problem_a_rotate_forward and show output
	problem_a_rotate_forward(input, angle);
	//Fill problem_b_rotate_backward and show output
	problem_b_rotate_backward(input, angle);
	//Fill problem_c_rotate_backward_interarea and show output
	problem_c_rotate_backward_interarea(input, angle);
}
