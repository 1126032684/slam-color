#include "looper.h"
using namespace cv;
using namespace rgbd_tutor;
cv::Mat FT(Mat &src){
	Mat Lab;
	cvtColor(src, Lab, CV_BGR2Lab); 
 
	int row=src.rows,col=src.cols;
 
	int Sal_org[row][col];
	memset(Sal_org,0,sizeof(Sal_org));
	
	Point3_<uchar>* p;
 
	int MeanL=0,Meana=0,Meanb=0;
	for (int i=0;i<row;i++){
		for (int j=0;j<col;j++){
			p=Lab.ptr<Point3_<uchar> > (i,j);
			MeanL+=p->x;
			Meana+=p->y;
			Meanb+=p->z;
		}
	}
	MeanL/=(row*col);
	Meana/=(row*col);
	Meanb/=(row*col);
 
	GaussianBlur(Lab,Lab,Size(3,3),0,0);
 
	Mat Sal=Mat::zeros(src.size(),CV_8UC1 );
 
	int val;
 
	int max_v=0;
	int min_v=1<<28;
 
	for (int i=0;i<row;i++){
		for (int j=0;j<col;j++){
			p=Lab.ptr<Point3_<uchar> > (i,j);
			val=sqrt( (MeanL - p->x)*(MeanL - p->x)+ (p->y - Meana)*(p->y-Meana) + (p->z - Meanb)*(p->z - Meanb) );
			Sal_org[i][j]=val;
			max_v=max(max_v,val);
			min_v=min(min_v,val);		
		}
	}
	
	cout<<max_v<<" "<<min_v<<endl;
	int X,Y;
    for (Y = 0; Y < row; Y++)
    {
        for (X = 0; X < col; X++)
        {
            Sal.at<uchar>(Y,X) = (Sal_org[Y][X] - min_v)*255/(max_v - min_v);        //    计算全图每个像素的显著性
        	//Sal.at<uchar>(Y,X) = (Dist[gray[Y][X]])*255/(max_gray);        //    计算全图每个像素的显著性
        }
    }
    return Sal;
}
double Entropy(Mat img) {
  double temp[256] = { 0.0 };
  for (int m = 0; m<img.rows; m++) 
  {
    const uchar* t = img.ptr<uchar>(m); 
    for (int n = 0; n<img.cols; n++) 
    {
      int i = t[n]; 
      temp[i] = temp[i] + 1; 
    }
  }
  for (int i = 0; i<256; i++)
  { 
    temp[i] = temp[i] / (img.rows*img.cols);
  } 
  double result = 0; 
  for (int i = 0; i<256; i++) 
  {
    if (temp[i] == 0.0) 
      result = result; 
    else 
      result = result - temp[i] * (log(temp[i]) / log(2.0));
  } 
  return result;
}
double ComEntropy(Mat img1, Mat img2, double img1_entropy, double img2_entropy) { 
  double temp[256][256] = { 0.0 };
  for (int m1 = 0, m2 = 0; m1 < img1.rows, m2 < img2.rows; m1++, m2++) 
  { 
    const uchar* t1 = img1.ptr<uchar>(m1); 
    const uchar* t2 = img2.ptr<uchar>(m2); 
    for (int n1 = 0, n2 = 0; n1 < img1.cols, n2 < img2.cols; n1++, n2++) 
    { 
      int i = t1[n1], j = t2[n2];
      temp[i][j] = temp[i][j] + 1;
    } 
  }
  for (int i = 0; i < 256; i++) 
  { 
    for (int j = 0; j < 256; j++)
    { 
      temp[i][j] = temp[i][j] / (img1.rows*img1.cols);
    }
  } 
  double result = 0.0; 
  for (int i = 0; i < 256; i++) 
  {
    for (int j = 0; j < 256; j++)
    { 
      if (temp[i][j] == 0.0) 
	result = result;
      else result = result - temp[i][j] * (log(temp[i][j]) / log(2.0));
    } 
  }
  img1_entropy = Entropy(img1); 
  img2_entropy = Entropy(img2);
  result = img1_entropy + img2_entropy - result; 
  return result;
}
vector<RGBDFrame::Ptr> Looper::getPossibleLoops( const RGBDFrame::Ptr& frame )
{
    vector<RGBDFrame::Ptr>  result;
    vector<double> H_values;
    Mat Src_ft=FT(frame->rgb);
    double Src_entry=Entropy(Src_ft);
    int th=frames.size()*0.4;
    for ( size_t i=0; i<frames.size(); i++ )
    {
        RGBDFrame::Ptr pf = frames[i];
	Mat loop_ft=FT(pf->rgb);
	double loop_entry=Entropy(loop_ft);
	double H_entry=ComEntropy(Src_ft,loop_ft,Src_entry,loop_entry);
	H_values.push_back(H_entry);
       /* double  score = vocab.score( frame->bowVec, pf->bowVec );
        if (score > min_sim_score && abs(pf->id-frame->id)>min_interval )
        {
            result.push_back( pf );
        }
        */
    }
    sort(H_values.begin(),H_values.end());
    for ( size_t i=0; i<frames.size(); i++ )
    {
        RGBDFrame::Ptr pf = frames[i];
	Mat loop_ft=FT(pf->rgb);
	double loop_entry=Entropy(loop_ft);
	double H_entry=ComEntropy(Src_ft,loop_ft,Src_entry,loop_entry);
	//H_values.push_back(H_entry);
        double  score = vocab.score( frame->bowVec, pf->bowVec );
        if (score > min_sim_score && abs(pf->id-frame->id)>min_interval&&H_entry>H_values[th] )
        {
            result.push_back( pf );
        }
    }
    return result;
}
