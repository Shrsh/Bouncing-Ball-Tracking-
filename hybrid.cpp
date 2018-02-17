#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/highgui.h>
#include <opencv2/highgui.hpp>
#include <opencv/cv.h>
#include<fstream>
#include<math.h>
#include <cmath>
#include <string>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include<time.h>
#include "/home/mudit/summer internship/hybrid-kalman-filter-bouncing ball/gnuplot_i.hpp"


// Namespace for using opencv objects.
using namespace cv;

// Namespace for using cout.
using namespace std;

#define DT 1.5
#define DTB 1.665
#define NUMBEROFSTATES 2
#define NUMBEROFMEASUREMENT 2
#define g 9.8
#define MASS 0.075
#define Friction 0.0

class ExtendedKalmanFilter
{
public:
    float** States;
    float** States_phi;
    float** ControlMatrix;
    float** StateTransition;
    float** Transition_phi;
    float** MeasurementFunction;
    float** Measurement;
    float** ProcessNoiseCovariance;
    float** MeasurementNoiseCovariance;
    float** Innovation;
    float** ErrorCovariance;
    float** Gain;
    float** function;
    float** temp1;
    float** temp2;
    float** temp3;
    float** temp4;
    float** temp5;
    float** temp6;
    float** temp7;
    float** z ;
    float** I ;
    float** IDEN;
    float** norm;
    float** one;
    float** two;
    float** three;
    float** four;
    float** five;
    float** six;
    float** x;
    bool IsBounce;
//    float** determinant;

    void InitializeFilter();
    void Filter(float *);
};


void mymakematrix(float **&temp, unsigned short int rws, unsigned short int cls) //create a matrix
{
    unsigned short int i;
    temp = new float* [rws];
    for(i=0; i<rws; ++i)
        temp[i] = new float [cls];
}

void mymakevector(float **&temp, unsigned short int rws) //create a matrix
{
    temp = new float* [rws];
}

void myaddmatrix(float **a, float **b, float **sum, unsigned short int rws, unsigned short int cls) //sum = a + b
{
    unsigned short int i, j;
    for (i=0; i<rws; ++i) // till all rows
        for (j=0; j<cls; ++j) //till all columns
            sum[i][j] = a[i][j] + b[i][j]; //add each eleent and store in sum matrix
}

void mysubmatrix(float **a, float **b, float **diff, unsigned short int rws, unsigned short int cls) //diff = a - b
{
    unsigned short int i, j;
    for (i=0; i<rws; ++i) //till all rows
        for (j=0; j<cls; ++j) //till all columns
            diff[i][j] = a[i][j] - b[i][j]; //subtract each element and store in diff matrix
}

void mytransposematrix(float **a, float **b, unsigned short int rws, unsigned short int cls ) //b = a'
{
    unsigned short int i, j;
    for (i=0; i<rws; ++i) //till all rows
        for (j=0; j<cls; ++j) //till all columns
            b[j][i] = a[i][j]; //do transpose
}

void mymulmatrix(float **a, float **b, float **mul, unsigned short int arws, unsigned short int acls, unsigned short int bcls) //b = a'
{
    float sum;
    unsigned short int i, j, k;
    for(i=0; i<arws; ++i) //till all rows
    {
        for(j=0; j<bcls; ++j) //till all columns
        {
            sum=0; //sum to zero
            for(k=0; k<acls; ++k) //just one row
                sum = sum + a[i][k]*b[k][j]; //multiplication
            mul[i][j] = sum; //accumulate sum
//            cout<<"sum is----"<<sum<<endl;
        }
    }
}

void mymulscalarmatrix(float **a, float** b, float mul, unsigned short int rws, unsigned short int cls) //b = a'
{
    unsigned short int i, j;
    for(i=0; i<rws; ++i) //till all rows
    {
        for(j=0; j<cls; ++j) //till all columns
        {
            b[i][j] = a[i][j]*mul;
        }
    }
}

void myprintmatrix(float **a, unsigned short int rws, unsigned short int cls) //b = a'
{
    unsigned short int i, j;
    for(i=0; i<rws; ++i) //till all rows
    {
        for(j=0; j<cls; ++j) //till all columns
        {
            std::cout<<a[i][j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<std::endl;
}

void mymakeidentitymatrix(float **a, unsigned short int rws, unsigned short int cls) //b = a'
{
    unsigned short int i, j;
    for(i=0; i<rws && i<cls; ++i) //till all rows
    {
        for(j=0; j<cls; ++j)
        {
            if(i==j)
                a[i][j] = 1;
            else
                a[i][j] = 0;
        }
    }
}

void mycholdecmatrix(float** a, float** l, unsigned short int n)
{
    float s;
    unsigned short int i, j, k;
    for(i=0; i<n; ++i)
    {
        for(j=0; j<(i+1); ++j)
        {
            s = 0;
            for (k=0; k<j; ++k)
                s += l[i][k] * l[j][k];
            if(i == j)
                l[i][j] = std::sqrt(a[i][i] - s);
            else
                l[i][j] = (a[i][j] - s)/l[j][j];
        }
    }
}

int sgn(double d)
{
    if(d>0)
        return 1;
    else if(d==0)
        return 0;
    else
        return -1;
}


void ExtendedKalmanFilter::InitializeFilter()
{

    mymakematrix(States, NUMBEROFSTATES, 1);
    mymakematrix(States_phi, NUMBEROFSTATES, 1);
    mymakematrix(ControlMatrix, NUMBEROFSTATES, 1);
    mymakematrix(StateTransition, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(Transition_phi, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(MeasurementFunction, NUMBEROFMEASUREMENT, NUMBEROFSTATES);
    mymakematrix(Measurement, NUMBEROFMEASUREMENT, 1);
    mymakematrix(ProcessNoiseCovariance, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(MeasurementNoiseCovariance, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);
    mymakematrix(ErrorCovariance, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(Innovation, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);
    mymakematrix(Gain, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
    mymakematrix(function, NUMBEROFSTATES,NUMBEROFMEASUREMENT);
    mymakematrix(temp1, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(temp2, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(temp3, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(temp4, NUMBEROFMEASUREMENT, 1);
    mymakematrix(temp5, NUMBEROFSTATES, 1);
    mymakematrix(temp6, NUMBEROFSTATES,NUMBEROFSTATES );
    mymakematrix(temp7, NUMBEROFMEASUREMENT, 1);
    mymakematrix(I, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT );
    mymakematrix(z, NUMBEROFMEASUREMENT, 1);
    mymakematrix(IDEN,NUMBEROFSTATES, NUMBEROFSTATES );
    mymakematrix(norm, NUMBEROFSTATES, 1);
    mymakematrix(one, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
    mymakematrix(two, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
    mymakematrix(three, NUMBEROFSTATES, NUMBEROFSTATES);
    mymakematrix(four, NUMBEROFMEASUREMENT,NUMBEROFMEASUREMENT );
    mymakematrix(five, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
    mymakematrix(six, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
    mymakematrix(x, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);


    IsBounce = false;
    States[0][0] = 0;
    States[1][0] = 0;

    ControlMatrix[1][0] = DT* g;
    ControlMatrix[0][0] = 0;

    IDEN[0][0]= 1;
    IDEN[1][1]= 1;


    ErrorCovariance[0][0] = 0.001;          //P
    ErrorCovariance[1][1] = 0.001;          //P

    ProcessNoiseCovariance[0][0] = 0.1;    //Q
    ProcessNoiseCovariance[1][1] = 0.1;   //Q


    MeasurementNoiseCovariance[0][0] = 0.001;        //R
    MeasurementNoiseCovariance[1][1] = 0.001;        //R


    StateTransition[0][0] = 1;           //F
    StateTransition[1][1] = 1;           //F
    StateTransition[0][1] = DT;          //F
    StateTransition[1][0] = 0;           //F

    temp7[0][0]=0;
    temp7[1][0]=0;


    MeasurementFunction[0][0] = 1;     //H
    MeasurementFunction[1][1] = 1;     //H

}


void ExtendedKalmanFilter::Filter(float measurement[])
{
//    temp7[0][0]=measurement[0];
//    temp7[1][0]=measurement[1];
    cout<<measurement[1]<<endl;

    //-----------transformation of measured values----------
    float k = sqrt((measurement[0]*measurement[0])+(measurement[1]*measurement[1]));
    temp7[0][0]= ((measurement[0]*measurement[0])-(measurement[1]*measurement[1]))/k;
    temp7[1][0]= (2*measurement[0]*measurement[1])/k;
    //-----------transformation of measured values----------


//    cout<<"----states intial---without kalman---  "<<States[0][0] <<"--- "<<States[1][0]<<endl;


    //----------States = F(x)*States + ControlMatrix-----------------------
    mymulmatrix(StateTransition, States, norm, NUMBEROFSTATES, NUMBEROFSTATES, 1 );
//    cout<<"---norm-----"<<norm[0][0]<<"---- "<<norm[1][0]<<endl;
    myaddmatrix(norm, ControlMatrix, States, NUMBEROFSTATES, 1);
    //----------States = F(x)*States + ControlMatrix-----------------------


    cout<<"----states at start------  "<<States[0][0] <<"--- "<<States[1][0]<<endl;


    //----------TRNASFORMATION OF predicted STATES------------------------
    float n = sqrt((States[0][0]*States[0][0])+(States[1][0]*States[1][0]));
    States_phi[1][0] = (2*States[0][0]*States[1][0])/n;
    States_phi[0][0] = ((States[0][0]*States[0][0])-(States[1][0]*States[1][0]))/n;
   //----------TRNASFORMATION OF predicted STATES------------------------


     cout<<"----states after transformation------  "<<States_phi[0][0] <<"--- "<<States_phi[1][0]<<endl;


     //---------------------STATE TRANSITION TRANSFORM-------------------
        float norm_y = pow(sqrt((States[0][0]*States[0][0]) + (States[1][0]*States[1][0])),3);
        Transition_phi[0][0] = ((States[0][0]*States[0][0]*States[0][0]) + (3*States[0][0]*(States[1][0]*States[1][0])))/norm_y;
        Transition_phi[1][1] = (2*(States[0][0]*States[0][0]*States[0][0]))/norm_y;
        Transition_phi[1][0]= (2*(States[1][0]*States[1][0]*States[1][0]))/norm_y;
        Transition_phi[0][1] = ((-3*States[1][0]*(States[0][0]*States[0][0])-(States[1][0]*States[1][0]*States[1][0])))/norm_y;
     //---------------------STATE TRANSITION TRANSFORM-------------------


//      cout<<"----transition_phi------"<<Transition_phi[0][0]<<"----"<<Transition_phi[1][1]<<"----"<<Transition_phi[1][0]<<"-----"<<Transition_phi[0][1]<<endl;


    //-------- P(k) = F(x)*P(k-1)*tr(F(x))+Q(k-1)
    mytransposematrix(Transition_phi, temp1, NUMBEROFSTATES, NUMBEROFSTATES);
//    cout<<"temp1 is---"<<temp1[0][0]<<"---"<<temp1[1][1]<<"-----"<<temp1[0][1]<<"-----"<<temp1[1][0]<<endl;


    mymulmatrix(Transition_phi, ErrorCovariance, temp2, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFSTATES);
//    cout<<"-----temp2 is---"<<temp2[0][0]<<"---"<<temp2[0][1]<<"------"<<temp2[1][0]<<"----"<<temp2[1][1]<<endl;

    mymulmatrix(temp2, temp1, temp3, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFSTATES);
//    cout<<"-----temp3 is---"<<temp3[0][0]<<"---"<<temp3[0][1]<<"------"<<temp3[1][0]<<"----"<<temp3[1][1]<<endl;

    myaddmatrix(temp3, ProcessNoiseCovariance, ErrorCovariance, NUMBEROFSTATES, NUMBEROFSTATES);
//    cout<<"error covariance  is---"<<ErrorCovariance[0][0]<<"---"<<ErrorCovariance[1][1]<<endl;
    //-------- P(k) = F(x)*P(k-1)*tr(F(x))+Q(k-1)



    //------------S(k) = H(x)*P(k)*tr(H(x))+R(k)
    mytransposematrix(MeasurementFunction, function, NUMBEROFMEASUREMENT, NUMBEROFSTATES);
    mymulmatrix(ErrorCovariance, function, two, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
    mymulmatrix(MeasurementFunction, two, Innovation, NUMBEROFMEASUREMENT, NUMBEROFSTATES, NUMBEROFMEASUREMENT);
    myaddmatrix(Innovation, MeasurementNoiseCovariance, Innovation, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);
    //------------S(k) = H(x)*P(k)*tr(H(x))+R(k)


    //--------INVERSE MATRIX CONVERSION STARTS--------------
    double determinant = (Innovation[0][0]*Innovation[1][1])-(Innovation[0][1]*Innovation[1][0]);
//    mytransposematrix(Innovation, two, NUMBEROFMEASUREMENT,NUMBEROFSTATES);
    four[0][0] = Innovation[1][1]/determinant;
    four[0][1] = -Innovation[0][1]/determinant;
    four[1][0] = -Innovation[1][0]/determinant;
    four[1][1] = Innovation[0][0]/determinant;

    //--------INVERSE MATRIX CONVERSION ENDS--------------




    //----------------KALMAN GAIN-------------
    mytransposematrix(MeasurementFunction, five, NUMBEROFMEASUREMENT, NUMBEROFSTATES);
//    cout<<"measurement jacobianT is---"<<five[0][0]<<"---"<<five[1][0]<<endl;
//    cout<<"measurement jacobianT is----"<<five[0][1]<<"---"<<five[1][1]<<"---"<<five[2][1]<<"---"<<five[3][1]<<"--"<<endl;

    mymulmatrix(ErrorCovariance, five, six, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFMEASUREMENT);

//    cout<<"six is---"<<six[0][0]<<"---"<<six[1][0]<<endl;
//    cout<<"error cov is----"<<six[0][1]<<"---"<<six[1][1]<<"---"<<six[2][1]<<"---"<<six[3][1]<<"--"<<endl;
    mymulmatrix(six, four , Gain, NUMBEROFSTATES, NUMBEROFMEASUREMENT, NUMBEROFMEASUREMENT);
    cout<<"kalman gain is "<<Gain[0][0]<<"---"<<Gain[1][0]<<endl;

    //------------------KALMAN GAIN----------------




    //-----------Y = LENGTH*CURRENT MEASUREMENT - LENGHT*PREVIOUSMEASUREMENT
    mymulmatrix(MeasurementFunction, States_phi, Measurement, NUMBEROFMEASUREMENT, NUMBEROFSTATES, 1);
    mysubmatrix(temp7, Measurement, temp4, NUMBEROFMEASUREMENT, 1);
    //------------------------------------------------------------------------



    //---------------PREDICTED STATES = PREVIOUS STATES + GAIN*Y
    mymulmatrix(Gain, temp4, temp5, NUMBEROFSTATES, NUMBEROFMEASUREMENT, 1);
    myaddmatrix(States_phi, temp5, States_phi, NUMBEROFSTATES, 1);
    //---------------------------------------------------------------------




    //---------------ERROR COVARIANCE = (IDEN - GAIN*MEASUREMENT JACOBIAN)*PREVIOUS ERROR COVARIANCE-------
    mymulmatrix(Gain, MeasurementFunction, three, NUMBEROFSTATES, NUMBEROFMEASUREMENT, NUMBEROFSTATES);
    mysubmatrix(IDEN, three, temp6, NUMBEROFSTATES, NUMBEROFSTATES);
    mymulmatrix(temp6, ErrorCovariance, ErrorCovariance, NUMBEROFSTATES, NUMBEROFSTATES, NUMBEROFSTATES);
//    cout<<"error covariance is---"<<ErrorCovariance[0][0]<<"---"<<ErrorCovariance[1][1]<<endl;


    //------------------------------------------------------------------------------------


        cout<<"----states--at---end---of transformation--  "<<States_phi[0][0] <<"--- "<<States_phi[1][0]<<endl;


    //-----------TAKING INVERSE TO GET THE REAL STATES BACK---------------
    float m = sqrt((States_phi[0][0]*States_phi[0][0]) + (States_phi[1][0]*States_phi[1][0]));
       States[1][0] = sgn(States_phi[1][0])*sqrt((0.5*m*(m-States_phi[0][0])));
       States[0][0] = sqrt(0.5*m*(m+States_phi[0][0]));
    //-----------TAKING INVERSE TO GET THE REAL STATES BACK---------------


       cout<<"----states--at---end  "<<States[0][0] <<"--- "<<States[1][0]<<endl;

}



int main()
{
    std::clock_t start;
    double duration;
    VideoCapture cap("/home/mudit/summer internship/hybrid-kalman-filter-bouncing ball/out.avi");
    if( !cap.isOpened()){
        cout << "Cannot open the video file" << endl;
        return -1;
    }

    int j = 0;
    Mat frame, imghsv0, imgG0, in1;

    double count = cap.get(CV_CAP_PROP_FRAME_COUNT); //get the frame count
    cout<<"FPS"<<count<<endl;
     cap.set(CV_CAP_PROP_AUTOFOCUS,count-1); //Set index to last frame
    namedWindow("MyVideo",CV_WINDOW_AUTOSIZE);
    int filter0[6] = {255, 6, 255, 9, 126, 70}; // Arrey for HSV filter, [Hmax, Hmin, Smax, Smin, Vmax, Vmin]
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS,cv::Size(3,3),cv::Point(-1,-1));
    int status = 0;
    vector<double> xdata, ydata, xreal, yreal, xin, yin, xout, yout;
    int i = 0;
    ExtendedKalmanFilter EKF;
    cout<<"sgh"<<endl;

    EKF.InitializeFilter();
    cout<<"FPS"<<endl;

    while(1)
    {
        start = std::clock();
        bool success = cap.read(frame);

      //  cout<<success<<endl;

        cap.retrieve(frame);

        if (!success)
        {
            cout << "Cannot read  frame " << endl;
            break;
        }



        if(1)
        {

            cv::cvtColor(frame, imghsv0, CV_BGR2HSV); // Convert colour to HSV

            cv::inRange(imghsv0, cv::Scalar(filter0[1], filter0[3], filter0[5]), cv::Scalar(filter0[0], filter0[2], filter0[4]), imgG0);

            cv::erode(imgG0,imgG0,kernel,cv::Point(-1,-1),2);
            cv::dilate(imgG0,imgG0,kernel,cv::Point(-1,-1),5);
            cv::erode(imgG0,imgG0,kernel,cv::Point(-1,-1),3);

            //         Find all contours
            std::vector<std::vector<cv::Point> > contours1;
            cv::findContours(imgG0.clone(), contours1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

//            contours1.clear();

            if(contours1.size()>=1)
            {
                //             Fill holes in each contour
                cv::drawContours(imgG0, contours1, -1, CV_RGB(255, 255, 255), 3);
                //                cout << contours1.size()<<endl;


                double avg_x1(0), avg_y1(0); // average of contour points


                for (int j = 0; j < contours1[0].size(); ++j)
                {
                    avg_x1 += contours1[0][j].x;
                    avg_y1 += contours1[0][j].y;

                }


                avg_x1 /= contours1[0].size();
                avg_y1 /= contours1[0].size();

                cv::circle( frame, cv::Point(avg_x1, avg_y1), 2, cv::Scalar(0, 0, 255), 8, 0 );



                // ------------------------------------------KALMAN FILTER------------------------------------



                float measurement[2], currpos[1], prevpos[1];

                currpos[0] = avg_y1;



                switch(status)
                {

                case 0:

                    EKF.States[0][0] = currpos[0];
                    prevpos[0] = currpos[0];

                    cout<<"states case0 "<<EKF.States[0][0] <<" "<<endl;

                    status++;
                    break;

                case 1:

                    EKF.States[0][0] = currpos[0];
                    prevpos[0] = currpos[0];
                    cout<<"----states case1----- "<<EKF.States[0][0]<<endl;
                    status++;
                    break;

                case 2:
                    if(i==16)
                    {
                    EKF.States[1][0] = (-currpos[0]+prevpos[0])/DT;
                    }
                    cout<<"states case2 "<<EKF.States[0][0] <<" "<<EKF.States[1][0]<<endl;
                    measurement[0] = currpos[0];       //new pos
                    measurement[1] = (currpos[0]-prevpos[0])/DT;          //new pos


                    EKF.Filter(measurement);


                    prevpos[0] = currpos[0];

                    xdata.push_back(i);
                    ydata.push_back(EKF.States[0][0]);
                    yreal.push_back(measurement[0]);
                    xreal.push_back(i);
                    xin.push_back(EKF.States[1][0]);
                    yin.push_back(i);
                    xout.push_back(i);
                    yout.push_back(measurement[1]);


                    double finalx1 = avg_x1;
                    double finaly1 = EKF.States[0][0];
//                    double velocity = EKF.States[0][1];
//                    cout<<finalx1<<endl;


                    cv::circle(frame, cv::Point(finalx1, finaly1), 2, cv::Scalar(255, 0, 0), 8, 0);

                    break;


                }
//                cv::imshow("bin", imgG0);
                cv::imshow("MyVideo", frame);
                cv::waitKey(200);


            }
            cout<<i<<endl;
            duration= (std::clock()-start)/(double) CLOCKS_PER_SEC;
//            cout<<"duration---"<<duration<<endl;
            i++;
        }
    }


    try
    {
        // Create Gnuplot object

        Gnuplot gp;
        Gnuplot pg;
        gp.set_terminal_std("jpeg");
        gp.set_title("velocity y-direction");
        pg.set_terminal_std("jpeg");
        pg.set_title("kalman y-position");

        // svg enhanced size 1000 1000 fname "Times" fsize 36;

//        set output "plot.svg"
//        set title "A simple plot of x^2 vs. x"
        // Configure the plot

        gp.set_style("lines");
        gp.set_xlabel("x");
        gp.set_ylabel("y");
        pg.set_style("lines");
        pg.set_xlabel("x");
        pg.set_ylabel("y");


        // Plot the data
        gp.plot_xy(xout, yout);
        gp.plot_xy(yin, xin);
        pg.plot_xy(xdata, ydata);
        pg.plot_xy(xreal, yreal);
        cout << "Press Enter to quit...";
        cin.get();

        return 0;
    }
    catch (const GnuplotException& error) {

        cerr << error.what() << endl;
        return 1;
    }
}


