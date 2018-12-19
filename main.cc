
#include <iostream>
#include <unistd.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define PROCESS_IMAGE
#define WINDOW_NAME "CVUI Hello World!"

#define CVUI_IMPLEMENTATION
#include "cvui.h"

#include "arfilter.h"

double drand()
{
    return (double)rand() / (double)(RAND_MAX);
}

cv::Mat getSobel( cv::Mat input )
{

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Generate grad_x and grad_y
    cv::Mat grad;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( input, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( input, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );

    /// Total Gradient (approximate)
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

    return grad;

}

void computeGradient( cv::Mat input, cv::Mat& grad_x_out, cv::Mat& grad_y_out )
{

    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Generate grad_x and grad_y
    cv::Mat grad;
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    /// Gradient X
    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
    Sobel( input, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
    cv::normalize( grad_x, grad_x_out, -1, 1, NORM_MINMAX, CV_32FC1);

    /// Gradient Y
    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
    Sobel( input, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
    cv::normalize( grad_y, grad_y_out, -1, 1, NORM_MINMAX, CV_32FC1);

    //    blur( grad_x, grad_x, cv::Size(45,45) );
    //    blur( grad_y, grad_y, cv::Size(45,45) );

}

double interp1( double instart, double instop, double outstart, double outstop, double in )
{
    double in_span = (instop - instart);
    double out_span = (outstop - outstart);
    double xx = in / in_span;
    return outstart + xx * out_span;
}

std::vector< cv::Point2f > columnToPoints( cv::Mat img, cv::Mat grad_x, cv::Mat grad_y, cv::Point startp, cv::Point endp, double pointSpeed, double propSpeed, double alphaFilter )
{

    ARFilter< double > arf( alphaFilter );
    LineIterator it(img, startp, endp, 8);
    std::vector< cv::Point2f > ret;

    Point2f curPos = Point2f( startp.x, startp.y );
    int k = 0;
    while( k < 1000 )
    {
        //        cerr << "curPos=" << curPos << endl;
        if( curPos.x < 0 || curPos.x > img.cols - 1 || curPos.y < 0 || curPos.y > img.rows - 1 )
            break;

        float gradVal_x = grad_x.at<float>( curPos );
        float gradVal_y = grad_y.at<float>( curPos );
        float cVal = img.at<uchar>( curPos ) / 255;
        float gradVal = max( max( gradVal_x, gradVal_y ), cVal );

        {
            double cPointSpeed = pointSpeed; // (cVal / 128.0) * pointSpeed;
            double ldist = ((double)startp.x - curPos.x);
            // cerr << "ldist=" << ldist << endl;
            curPos.x += ldist * propSpeed;
            curPos.x += (1.0 * gradVal) * cPointSpeed;
            curPos.x = arf.update( curPos.x );
//            curPos.x += -2 + drand() * 4.0;
        }
        curPos.y += 1.0; // - gradVal_y;

        if( !( curPos.x < 0 || curPos.x > img.cols - 1 || curPos.y < 0 || curPos.y > img.rows - 1 ) )
            ret.push_back( curPos );
        ++k;
    }

    return ret;
}

int main( int argc, char** argv )
{

    srand(time(NULL));

#ifndef PROCESS_IMAGE
    namedWindow( "frame" );
    cv::VideoCapture cap( argv[1] );
    bool isPlaying = true;

    cv::Mat frame;

    for(;;)
    {
        if( isPlaying )
        {
            cap >> frame;
            if( frame.cols <= 0 || frame.rows <= 0 )
                continue;

        }

        imshow("frame", frame);

        char c = cv::waitKey(5);
        if( c == ' ' )
        {
            isPlaying = !isPlaying;
        }
        else if( c == 'q' )
        {
            break;
        }

    }
#else
    cv::Mat input = cv::imread( argv[1] );

    // Create a frame where components will be rendered to.
    cv::Mat frame = cv::Mat(200, 500, CV_8UC3);

    // Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
    cvui::init(WINDOW_NAME);

    int stepx = 10;
    double pointSpeed = 10.0;
    double propSpeed = 0.01;
    double alphaFilter = 0.2;

    bool needUpdate = false;

    cv::Mat frameg;
    cv::Mat frameout = cv::Mat::zeros( input.rows, input.cols, CV_8UC1 );
    cvtColor( input, frameg, CV_RGB2GRAY );

    cv::Mat grad_x, grad_y;
    computeGradient( frameg, grad_x, grad_y );

    while(true)
    {

        // Fill the frame with a nice color
        frame = cv::Scalar(49, 52, 49);

        // Render UI components to the frame
        if( cvui::trackbar(frame, 5, 10, 240, &stepx, (int)1., (int)100.) )
        {
            needUpdate = true;
        }

        if( cvui::trackbar(frame, 5, 60, 240, &pointSpeed, 0., 100.) )
        {
            needUpdate = true;
        }

        if( cvui::trackbar(frame, 5, 110, 240, &propSpeed, 0., 1.) )
        {
            needUpdate = true;
        }

        if( cvui::trackbar(frame, 5, 160, 240, &alphaFilter, 0., 1.) )
        {
            needUpdate = true;
        }

        if (cvui::button(frame, 300, 80, "&Quit")) {
            break;
        }

        double ratio = 0.1;
        if( true )
        {
            cerr << "need update pointSpeed=" << pointSpeed << " stepx=" << stepx << endl;
            frameout = cv::Mat::zeros( input.rows, input.cols, CV_32FC1 );
            for( int i = 0; i < frameg.cols; i += stepx )
            {
                cv::Mat frameout_temp = cv::Mat::zeros( input.rows, input.cols, CV_32FC1 );
                std::vector< Point2f > pret = columnToPoints( frameg, grad_x, grad_y, Point(i,0), Point(i,frameg.rows-1), pointSpeed, propSpeed, alphaFilter );
                std::vector< Point2f > pret2 = pret;
                // cerr << "pret.Size()=" << pret.size() << endl;
                //Option 1: use polylines
                Mat curve(pret2, true);
                curve.convertTo(curve, CV_32S); //adapt type for polylines
                int dcenter_v = abs( i - frameg.cols / 2 );
                polylines(frameout_temp, curve, false, Scalar( interp1( 0, frameg.cols / 2, 1.0, 0.25, dcenter_v ) ), 1, CV_AA);
                frameout += frameout_temp;
            }
            needUpdate = false;
            cv::normalize( frameout, frameout, 0, 255, NORM_MINMAX, CV_8UC1 );
        }

        // Update cvui stuff and show everything on the screen
        cvui::imshow(WINDOW_NAME, frame);

        imshow( "frame", frameout );
        cv::waitKey(20);

    }

#endif

    return 0;

}

