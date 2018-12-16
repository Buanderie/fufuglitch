
#include <iostream>
#include <unistd.h>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define PROCESS_IMAGE

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

double interp1( double instart, double instop, double outstart, double outstop, double in )
{
    double in_span = (instop - instart);
    double out_span = (outstop - outstart);
    double xx = in / in_span;
    return outstart + xx * out_span;
}

std::vector< cv::Point > columnToPoints( cv::Mat img, cv::Point startp, cv::Point endp, int space )
{
    const int minSpace = space / 4;
    const int maxSpace = space;

    int curSpace = minSpace;
    int curSum = 0;

    LineIterator it(img, startp, endp, 8);
    std::vector< cv::Point > ret;

    int k = 0;
    for(int i = 0; i < it.count; i++, ++it)
    {
        Point pt= it.pos();
        uchar curval = (uchar)(**it);
        // cerr << "curval=" << (int)curval << endl;
        if( k >= curSpace )
        {
            // output point
            cv::Point np( startp.x, i );
            ret.push_back( np );
            // cerr << "np=" << np << endl;
            // Compute new space
            curSpace = interp1( 0, 255, minSpace, maxSpace, (double)curSum / (double)curSpace );
            curSum = 0;
            k = 0;
        }
        ++k;
        curSum += 255 - (int)curval;
    }

    return ret;
}

double drand()
{
    return (double)rand() / (double)(RAND_MAX);
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
    cv::Mat frame = cv::imread( argv[1] );

    while(true)
    {
        cv::Mat frameg;
        cv::Mat frameout = cv::Mat::zeros( frame.rows, frame.cols, CV_8UC1 );

        cvtColor( frame, frameg, CV_RGB2GRAY );
        cv::Mat sobimg = frameg.clone();
        blur( sobimg, sobimg, cv::Size(37,37) );
        imshow( "sob", sobimg );

        int spacex = 10;
        int spacey = 10;

        int iv_noise_max = 20;

        for( int i = 0; i < frameg.cols; i += spacex )
        {
            std::vector< Point > res = columnToPoints( frameg, Point(i,0), Point(i,frameg.rows), spacey );
            for( Point p : res )
            {

                // Add noise according to intensity value ?
                int iiv = (int)sobimg.at<uchar>( p );
                int iv = 255 - iiv;
                int x_noise = interp1( 0, 255, 1, iv_noise_max, iv ) * drand();
                int y_noise = interp1( 0, 255, 1, iv_noise_max, iv ) * drand();

                int x_offset = 0;
                if( rand() % 4 == 0 )
                    x_offset = -frameg.cols / 2 + rand() % frameg.cols * 2;

                p.x += x_noise;
                p.y += y_noise + x_offset;

                // cv::circle( frameout, p, 3, Scalar(iiv,iiv,iiv) );
                const int linew = rand() % 20 + 2;
                for( int k = -linew / 2; k <= linew; ++k )
                {
                    Point pp = p;
    //                if( rand() % 2 == 0 )
                        p.x += k;
    //                else
    //                    p.x += k;
                    if( pp.x >= 0 && pp.x < frameout.cols && pp.y >= 0 && pp.y < frameout.rows )
                    {
                        int iivl = (int)frameg.at<uchar>( p );
                        frameout.at<uchar>( pp ) = (1 * iiv + 0 * iivl);
                    }
                }
            }
        }

        cv::Mat frameout_blur;
        cv::Mat result;

        /*
        int blursize = max(spacex, spacey) * 2;
        if(blursize % 2 == 0 )
            blursize++;
        GaussianBlur( frameout, frameout_blur, Size(blursize,blursize), 0 );
        */

        double pointRatio = 0.80;
        cv::addWeighted( frameout, pointRatio, sobimg, 1.0 - pointRatio, 0, result, CV_8UC3 );
        cv::normalize( result, result, 0, 255, NORM_MINMAX, CV_8UC1);

        imshow( "frame", frameg );
        imshow( "out", result );

        imwrite( "/tmp/out.png", result );

        cv::waitKey(5);
    }

#endif

    return 0;

}

