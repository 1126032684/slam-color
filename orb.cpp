/*************************************************************************
	> File Name: orb.cpp
	> Author: 
	> Mail: 
	> Created Time: 2016年02月29日 星期一 12时14分06秒
 ************************************************************************/

#include <iostream>
#include "common_headers.h"
#include "converter.h"
#include "orb.h"
using namespace std;
using namespace cv;
using namespace rgbd_tutor;

vector<cv::DMatch> OrbFeature::match( const RGBDFrame::Ptr& frame1, const RGBDFrame::Ptr& frame2 ) const
{
    vector<double> ss;
    vector<cv::KeyPoint> kps1,kps2;
    vector< vector<cv::DMatch> > matches_knn;
    kps1=frame1->getAllKeypoints();
    kps2=frame2->getAllKeypoints();
    cv::Mat desp1 = frame1->getAllDescriptors();
    cv::Mat desp2 = frame2->getAllDescriptors();
    matcher->knnMatch( desp1, desp2, matches_knn, 2 );
    vector< cv::DMatch > matches,end_matches;
    for ( size_t i=0; i<matches_knn.size(); i++ )
    {
        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
	{
	    matches.push_back( matches_knn[i][0] );
	}
    }
   // cout<<"match size is &&&&&&&&&&&&&&&&"<<matches.size()<<endl;
    for(size_t i=0;i<matches.size();i++)
    {
      double b1,b2,g1,g2,r1,r2;
    //  Vec3b pix1 = frame1->rgb.at<Vec3b>(kps1[matches[i].queryIdx].pt.x,kps1[matches[i].queryIdx].pt.y);
   //   Vec3b pix2 = frame2->rgb.at<Vec3b>(kps2[matches[i].queryIdx].pt.x,kps2[matches[i].queryIdx].pt.y);
      b1=frame1->rgb.at<Vec3b>(kps1[matches[i].queryIdx].pt.y,kps1[matches[i].queryIdx].pt.x)[0];//B
       b2=frame2->rgb.at<Vec3b>(kps2[matches[i].trainIdx].pt.y,kps2[matches[i].trainIdx].pt.x)[0];
       g1=frame1->rgb.at<Vec3b>(kps1[matches[i].queryIdx].pt.y,kps1[matches[i].queryIdx].pt.x)[1];//G
       g2=frame2->rgb.at<Vec3b>(kps2[matches[i].trainIdx].pt.y,kps2[matches[i].trainIdx].pt.x)[1];
       r1=frame1->rgb.at<Vec3b>(kps1[matches[i].queryIdx].pt.y,kps1[matches[i].queryIdx].pt.x)[2];//R
     r2=frame2->rgb.at<Vec3b>(kps2[matches[i].trainIdx].pt.y,kps2[matches[i].trainIdx].pt.x)[2];
    //  uchar b1=pix1[0];
   //   uchar b2=pix2[0];
//       uchar g1=pix1[1];
//       uchar g2=pix2[1];
//       uchar r1=pix1[2];
//       uchar r2=pix2[2];
      double sum,t;
     // cout<<"b1  :"<<b1<<"	b2 is :"<<b2<<endl;
     //  cout<<"g1  :"<<g1<<"	g2 is :"<<g2<<endl;
   //     cout<<"r1  :"<<r1<<"	r2 is :"<<r2<<endl;
       sum=(b2-b1)*(b2-b1)+(g2-g1)*(g2-g1)+(r2-r1)*(r2-r1);
   //    cout<<"sum is .............."<<sum<<endl;
    //  sum=((double)b2-double(b1))*((double)b2-double(b1))+((double)g2-(double)g1)*((double)g2-(double)g1)+((double)r2-(double)r1)*((double)r2-(double)r1);
      t=sqrt(sum);
    //  cout<<"t is ,,,,,,,,,,,,,,"<<t<<endl;
      ss.push_back(t);
   
    }
    sort(ss.begin(),ss.end());
    int th=matches.size()*0.5;
    double res=ss[th];
    for(size_t i=0;i<matches.size();i++)
    {
            double b1,b2,g1,g2,r1,r2;
       b1=frame1->rgb.at<Vec3b>(kps1[matches[i].queryIdx].pt.y,kps1[matches[i].queryIdx].pt.x)[0];//B
       b2=frame2->rgb.at<Vec3b>(kps2[matches[i].trainIdx].pt.y,kps2[matches[i].trainIdx].pt.x)[0];
       g1=frame1->rgb.at<Vec3b>(kps1[matches[i].queryIdx].pt.y,kps1[matches[i].queryIdx].pt.x)[1];//G
       g2=frame2->rgb.at<Vec3b>(kps2[matches[i].trainIdx].pt.y,kps2[matches[i].trainIdx].pt.x)[1];
       r1=frame1->rgb.at<Vec3b>(kps1[matches[i].queryIdx].pt.y,kps1[matches[i].queryIdx].pt.x)[2];//R
       r2=frame2->rgb.at<Vec3b>(kps2[matches[i].trainIdx].pt.y,kps2[matches[i].trainIdx].pt.x)[2];
       double sum,t;
       sum=(b2-b1)*(b2-b1)+(g2-g1)*(g2-g1)+(r2-r1)*(r2-r1);
       t=sqrt(sum);
         if(t<res)
      {
	end_matches.push_back(matches[i]);
      }
    }
 //  return matches;
    return end_matches;
}
