#include "mllmnn.h"
#include <highgui.h>

int main()
{
	int totaltest = 100;
	/* read image file */
	FILE* test = fopen( "t10k-images.idx3-ubyte", "r" );
	char bigendian[4];
	int a;
	fread( bigendian, 1, 4, test );
	a = (bigendian[0]<<24)+(bigendian[1]<<16)+(bigendian[2]<<8)+bigendian[3];
	unsigned int count;
	fread( bigendian, 1, 4, test );
	count = (bigendian[0]<<24)+(bigendian[1]<<16)+(bigendian[2]<<8)+bigendian[3];
	unsigned int width, height;
	fread( bigendian, 1, 4, test );
	width = (bigendian[0]<<24)+(bigendian[1]<<16)+(bigendian[2]<<8)+bigendian[3];
	fread( bigendian, 1, 4, test );
	height = (bigendian[0]<<24)+(bigendian[1]<<16)+(bigendian[2]<<8)+bigendian[3];
	printf("%d %d %d %d\n", a, count, width, height);
	unsigned char* image = (unsigned char*)malloc(width*height);
	CvMat* train_data = cvCreateMat( totaltest, width*height, CV_64FC1 );
	double* data_vec = train_data->data.db;
	for ( int i = 0; i < totaltest; i++ )
	{
		fread( image, 1, width*height, test );
		for ( int j = 0; j < width*height; j++ )
		{
			*data_vec = (double)image[j]/255.;
			data_vec++;
		}
	}
	fclose( test );
	FILE* label = fopen( "t10k-labels.idx1-ubyte", "r" );
	fread( bigendian, 1, 4, label );
	int la = (bigendian[0]<<24)+(bigendian[1]<<16)+(bigendian[2]<<8)+bigendian[3];
	fread( bigendian, 1, 4, label );
	unsigned int lcount = (bigendian[0]<<24)+(bigendian[1]<<16)+(bigendian[2]<<8)+bigendian[3];
	CvMat* responses = cvCreateMat( totaltest, 1, CV_32SC1 );
	int* rv = responses->data.i;
	CvMat* sample_idx = cvCreateMat( totaltest, 1, CV_32SC1 );
	int* si = sample_idx->data.i;
	CvMemStorage* storage = cvCreateMemStorage(0);
	CvSeq* cls[10];
	for ( int i = 0; i < 10; i++ )
		cls[i] = cvCreateSeq( 0, sizeof(CvSeq), sizeof(int), storage );
	for ( int i = 0; i < totaltest; i++, rv++, si++ )
	{
		unsigned char flag;
		fread( &flag, 1, 1, label );
		*si = i;
		*rv = flag;
		cvSeqPush(cls[flag], &i);
	}
	fclose( label );
	printf("%d %d\n", la, lcount);
	CvLMNNParams params = CvLMNNParams(100);
	int size[] = {params.maxidx, params.maxidx};
	CvSparseMat* target_neighbors = cvCreateSparseMat( 2, size, CV_32SC1 );
	int top3i[3];
	double top3v[3];
	for ( int k = 0; k < 10; k++ )
	{
		for ( int i = 0; i < cls[k]->total; i++ )
		{
			int idx1 = *CV_GET_SEQ_ELEM( int, cls[k], i );
			CvMat idx1data = cvMat( width*height, 1, CV_64FC1, train_data->data.db+width*height*idx1 );
			top3i[0] = 0; top3i[1] = 0; top3i[2] = 0;
			top3v[0] = 1e5; top3v[1] = 1e5; top3v[2] = 1e5;
			for ( int j = 0; j < cls[k]->total; j++ )
				if ( i != j )
				{
					int idx2 = *CV_GET_SEQ_ELEM( int, cls[k], j );
					CvMat idx2data = cvMat( width*height, 1, CV_64FC1, train_data->data.db+width*height*idx2 );
					double norm = cvNorm( &idx1data, &idx2data );
					if ( norm < top3v[2] )
					{
						top3v[2] = norm;
						top3i[2] = idx2;
					}
					for ( int t = 2; t > 0; t-- )
					{
						if ( top3v[t] < top3v[t-1] )
						{
							double tv = top3v[t-1];
							int ti = top3i[t-1];
							top3v[t-1] = top3v[t];
							top3i[t-1] = top3i[t];
							top3v[t] = tv;
							top3i[t] = ti;
						}
					}
				}
			for ( int t = 0; t < 3; t++ )
				if ( top3v[t] < 1e5 )
					cvSetReal2D( target_neighbors, idx1, top3i[t], 1 );
		}
	}
	printf("target neighbors found\n");
	CvSparseMat* negatives = cvCreateSparseMat( 2, size, CV_32SC1 );
	cvSetReal2D( negatives, 1, 0, 1 );
	cvSetReal2D( negatives, 0, 1, 1 );
	cvSetReal2D( negatives, 2, 3, 1 );
	cvSetReal2D( negatives, 3, 2, 1 );
	cvSetReal2D( negatives, 3, 8, 1 );
	cvSetReal2D( negatives, 8, 3, 1 );
	cvSetReal2D( negatives, 8, 6, 1 );
	cvSetReal2D( negatives, 6, 8, 1 );
	CvLMNN* lmnn = new CvLMNN(params);
	lmnn->train( train_data, responses, sample_idx, target_neighbors, negatives );
	lmnn->save("lmnndata");
	lmnn->load("lmnndata");

	CvMat* testcase = cvCreateMat( width*height, 1, CV_64FC1 );
	double* test_vec = testcase->data.db;
	data_vec = train_data->data.db+20*width*height;
	for ( int i = 0; i < width*height; i++ )
	{
		*test_vec = *data_vec;
		test_vec++;
		data_vec++;
	}
	IplImage* img = cvCreateImage( cvSize(width, height), 8, 1 );
	unsigned char* img_vec = (unsigned char*)img->imageData;
	test_vec = testcase->data.db;
	for ( int i = 0; i < width*height; i++ )
	{
		*img_vec = (unsigned char)(*test_vec*255.0);
		test_vec++;
		img_vec++;
	}
	cvSaveImage( "original.png", img );
	CvMat* abst = lmnn->abstract( testcase );
	CvMat* result = lmnn->reconstruct( abst );
	IplImage* rimg = cvCreateImage( cvSize(width, height), 8, 1 );
	unsigned char* rimg_vec = (unsigned char*)rimg->imageData;
	double* result_vec = result->data.db;
	for ( int i = 0; i < width*height; i++ )
	{
		*rimg_vec = (unsigned char)(*result_vec*255.0);
		result_vec++;
		rimg_vec++;
	}
	cvSaveImage( "result.png", rimg );
	delete lmnn;
	return 0;
}
