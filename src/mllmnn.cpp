#include "mllmnn.h"

struct CvVector
{
	int idx;
	union {
		uchar* ptr;
		int* i;
		float* fl;
		double* db;
	} data;
};

CvVector cvVector( int idx, void* ptr )
{
	CvVector vec;
	vec.idx = idx;
	vec.data.ptr = (uchar*)ptr;
	return vec;
}

class CvCategoricalClasses
{
	private:
		int _capacity;
		std::map<int, CvSeq*>* hash;
	public:
		CvCategoricalClasses(int capacity);
		void add(int cls, CvSeq* seq);
		void remove(int cls);
		void clear();
		CvSeq* get(int cls);
		~CvCategoricalClasses();
};

CvCategoricalClasses::CvCategoricalClasses(int capacity)
 : _capacity(capacity)
{
	hash = new std::map<int, CvSeq*> [capacity];
}

void CvCategoricalClasses::add(int cls, CvSeq* seq)
{
	hash[cls%_capacity][cls] = seq;
}

void CvCategoricalClasses::remove(int cls)
{
	if (hash[cls%_capacity].count(cls) > 0)
		hash[cls%_capacity].erase(cls);
}

void CvCategoricalClasses::clear()
{
	for ( int i = 0; i < _capacity; i++ )
		hash[i].clear();
}

CvSeq* CvCategoricalClasses::get(int cls)
{
	if ( hash[cls%_capacity].count(cls) > 0 )
		return hash[cls%_capacity][cls];
	return 0;
}

CvCategoricalClasses::~CvCategoricalClasses()
{
	delete [] hash;
}

CvLMNN::CvLMNN( const CvLMNNParams _params )
: params(_params)
{
	cgclass = new CvCategoricalClasses(512);
	storage = cvCreateMemStorage(0);
	clsidx = cvCreateSeq( 0, sizeof(CvSeq), sizeof(int), storage );
	updated = false;
	rebuilt = false;
}

CvLMNN::~CvLMNN()
{
	if ( rebuilt )
		clear();
	if ( L != NULL )
		cvReleaseMat( &L );
	delete cgclass;
	cvReleaseMemStorage(&storage);
}

void CvLMNN::clear()
{
	cvReleaseMemStorage( &active_set_storage );
	cvReleaseSparseMat( &target_neighbors );
	cvReleaseSparseMat( &negatives );
	cvReleaseSparseMat( &N );
	cvReleaseMat( &Lp );
	cvReleaseMat( &L );
	cvReleaseMat( &tL );
	cvReleaseMat( &dL );
	cvReleaseMat( &G );
	cvReleaseMat( &Cij );
	cvReleaseMat( &Cl );
	cvReleaseMat( &C );
	cgclass->clear();
	cvReleaseMemStorage( &cc_storage );
	cvClearSeq( clsidx );
	rebuilt = false;
}

void CvLMNN::rebuild_cache()
{
	C = cvCreateMat( var_count, var_count, CV_64FC1 );
	Cl = cvCreateMat( var_count, var_count, CV_64FC1 );
	Cij = cvCreateMat( var_count, 1, CV_64FC1 );
	G = cvCreateMat( var_count, var_count, CV_64FC1 );
	dL = cvCreateMat( params.dims, var_count, CV_64FC1 );
	tL = cvCreateMat( params.dims, var_count, CV_64FC1 );
	L = cvCreateMat( params.dims, var_count, CV_64FC1 );
	Lp = cvCreateMat( params.dims, 1, CV_64FC1 );
	int size[] = {params.maxidx, params.maxidx, params.maxidx};
	N = cvCreateSparseMat( 3, size, CV_32SC1 );
	target_neighbors = cvCreateSparseMat( 2, size, CV_32SC1 );
	negatives = cvCreateSparseMat( 2, size, CV_32SC1 );
	use_negatives = false;
	cc_storage = cvCreateChildMemStorage( storage );
	active_set_storage = cvCreateChildMemStorage( storage );
	active_set_child_storage = cvCreateChildMemStorage( active_set_storage );
	active_set = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSeq*), active_set_storage );
	CvRNG rng_state = cvRNG(0xdeadbeef);
	cvRandArr( &rng_state, L, CV_RAND_NORMAL, cvRealScalar(0), cvRealScalar(0.1) );
	cvCopy( L, tL );
	rebuilt = true;
}

static inline double norm2( const CvMat* arr )
{
	double ret = 0;
	double* s = arr->data.db;
	for ( int i = 0; i < arr->rows*arr->cols; i++, s++ )
		ret += *s * *s;
	return ret;
}

void CvLMNN::rebuild_active_set()
{
	cvReleaseSparseMat( &N );
	int size[] = {params.maxidx, params.maxidx, params.maxidx};
	N = cvCreateSparseMat( 3, size, CV_8UC1 );
	cvReleaseMemStorage( &active_set_child_storage );
	active_set_child_storage = cvCreateChildMemStorage( active_set_storage );
	cvClearSeq( active_set );
	for ( int k = 0; k < clsidx->total; k++ )
	{
		int cls = *CV_GET_SEQ_ELEM( int, clsidx, k );
		CvSeq* vecs = cgclass->get( cls );
		if ( vecs != 0 )
		{
			for ( int i = 0; i < vecs->total; i++ )
			{
				CvVector* vec_i = CV_GET_SEQ_ELEM( CvVector, vecs, i );
				CvMat Cihdr = cvMat( var_count, 1, CV_64FC1, vec_i->data.db );
				for ( int j = i+1; j < vecs->total; j++ )
				{
					CvVector* vec_j = CV_GET_SEQ_ELEM( CvVector, vecs, j );
					int ijf = (int)cvGetReal2D( target_neighbors, vec_i->idx, vec_j->idx );
					if ( ijf != 0 )
					{
						CvSeq* set = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvVector), active_set_child_storage );
						cvSeqPush( active_set, &set );
					}
				}
			}
		}
	}
}

bool CvLMNN::expand_active_set()
{
	bool expanded = false;
	int tk = 0;
	for ( int k = 0; k < clsidx->total; k++ )
	{
		int cls = *CV_GET_SEQ_ELEM( int, clsidx, k );
		CvSeq* vecs = cgclass->get( cls );
		if ( vecs != 0 )
		{
			for ( int i = 0; i < vecs->total; i++ )
			{
				CvVector* vec_i = CV_GET_SEQ_ELEM( CvVector, vecs, i );
				CvMat Cihdr = cvMat( var_count, 1, CV_64FC1, vec_i->data.db );
				for ( int j = i+1; j < vecs->total; j++ )
				{
					CvVector* vec_j = CV_GET_SEQ_ELEM( CvVector, vecs, j );
					int ijf = (int)cvGetReal2D( target_neighbors, vec_i->idx, vec_j->idx );
					if ( ijf != 0 )
					{
						CvMat Cjhdr = cvMat( var_count, 1, CV_64FC1, vec_j->data.db );
						cvSub( &Cihdr, &Cjhdr, Cij );
						cvMatMul( tL, Cij, Lp );
						double cijn_1 = norm2( Lp )+params.margin;
						double cijn_2 = cijn_1+params.marginblur;
						CvSeq* set = *CV_GET_SEQ_ELEM( CvSeq*, active_set, tk );
						tk++;
						for ( int fk = 0; fk < clsidx->total; fk++ )
						{
							int fcls = *CV_GET_SEQ_ELEM( int, clsidx, fk );
							if ( fcls != cls && ( !use_negatives || (int)cvGetReal2D( negatives, cls, fcls ) != 0 ) )
							{
								CvSeq* vecl = cgclass->get( fcls );
								if ( vecl != 0 )
								{
									for ( int l = 0; l < vecl->total; l++ )
									{
										CvVector* vec_l = CV_GET_SEQ_ELEM( CvVector, vecl, l );
										CvMat Clhdr = cvMat( var_count, 1, CV_64FC1, vec_l->data.db );
										cvSub( &Cihdr, &Clhdr, Cij );
										cvMatMul( tL, Cij, Lp );
										double ciln = norm2( Lp );
										if ( ciln < cijn_2 )
										{
											cvSeqPush( set, vec_l );
											if ( ciln < cijn_1 )
											{
												int flag = (int)cvGetReal3D( N, vec_i->idx, vec_j->idx, vec_l->idx );
												if ( !flag )
													expanded = true;
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
	return expanded;
}

void CvLMNN::trainLMNN()
{
	cvZero( G );

	printf("rebuilding active set...\n");
	rebuild_active_set();
	printf("expanding active set...\n");
	expand_active_set();
	
	double stepsize = params.stepsize;
	double best = 0;
	int ct = 0;
	bool reinitialize = true;
	for ( int t = 0; t < params.maxiter; t++ )
	{
		int tk = 0;
		double tbest = 0;
		for ( int k = 0; k < clsidx->total; k++ )
		{
			int cls = *CV_GET_SEQ_ELEM( int, clsidx, k );
			CvSeq* vecs = cgclass->get( cls );
			if ( vecs != 0 )
			{
				for ( int i = 0; i < vecs->total; i++ )
				{
					CvVector* vec_i = CV_GET_SEQ_ELEM( CvVector, vecs, i );
					CvMat Cihdr = cvMat( var_count, 1, CV_64FC1, vec_i->data.db );
					for ( int j = i+1; j < vecs->total; j++ )
					{
						CvVector* vec_j = CV_GET_SEQ_ELEM( CvVector, vecs, j );
						int ijf = (int)cvGetReal2D( target_neighbors, vec_i->idx, vec_j->idx );
						if ( ijf != 0 )
						{
							CvMat Cjhdr = cvMat( var_count, 1, CV_64FC1, vec_j->data.db );
							cvSub( &Cihdr, &Cjhdr, Cij );
							cvMatMul( tL, Cij, Lp );
							cvMulTransposed( Cij, C, 0 );
							cvAdd( C, G, G );
							double cijn_1 = norm2( Lp );
							tbest += cijn_1;
							cijn_1 += params.margin;
							CvSeq* set = *CV_GET_SEQ_ELEM( CvSeq*, active_set, tk );
							tk++;
							for ( int l = 0; l < set->total; l++ )
							{
								CvVector* vec_l = CV_GET_SEQ_ELEM( CvVector, set, l );
								CvMat Clhdr = cvMat( var_count, 1, CV_64FC1, vec_l->data.db );
								cvSub( &Cihdr, &Clhdr, Cij );
								cvMatMul( tL, Cij, Lp );
								double ciln = norm2( Lp );
								int flag = (int)cvGetReal3D( N, vec_i->idx, vec_j->idx, vec_l->idx );
								if ( ciln < cijn_1 )
								{
									tbest += params.u*(cijn_1-ciln);
									if ( !flag )
									{
										cvMulTransposed( Cij, Cl, 0 );
										cvSub( C, Cl, Cl );
										cvScaleAdd( Cl, cvRealScalar(params.u), G, G );
										cvSetReal3D( N, vec_i->idx, vec_j->idx, vec_l->idx, 1 );
									}
								} else if ( flag ) {
									cvMulTransposed( Cij, Cl, 0 );
									cvSub( C, Cl, Cl );
									cvScaleAdd( Cl, cvRealScalar(-params.u), G, G );
									cvSetReal3D( N, vec_i->idx, vec_j->idx, vec_l->idx, 0 );
								}
							}
						}
					}
				}
			}
		}
		double oldbest = best;
		if ( tbest < best || reinitialize )
		{
			best = tbest;
			cvCopy( tL, L );
			reinitialize = false;
		} else if ( !reinitialize ) {
			stepsize *= .5;
			if ( stepsize < params.minstepsize )
				stepsize = params.minstepsize;
			printf("rescale stepsize to: %f\n", stepsize);
		}
		printf("round %d, optimized: %f, current best: %f\n", t, tbest, best);
		if ( (t+1)%params.reinitialize == 0 )
		{
				printf("expanding active set...\n");
				expand_active_set();
				reinitialize = true;
				continue;
		}
		if ( fabs(best-oldbest) < params.epsilon )
		{
			ct++;
			if ( ct > params.maxconverge && !reinitialize )
			{
				printf("expanding active set...\n");
				if (!expand_active_set() )
				{
					printf("met criteria, early quit...\n");
					break;
				}
				continue;
			}
		} else
			ct = 0;
		cvGEMM( tL, G, 2., 0, 0, dL, 0 );
		cvScaleAdd( dL, cvRealScalar(-stepsize), tL, tL );
	}
}

bool CvLMNN::train( const CvMat* _train_data, const CvMat* _responses,
					const CvMat* _sample_idx, const CvSparseMat* _target_neighbors,
					const CvSparseMat* _negatives, bool update_base )
{
	bool ok = false;
	CV_FUNCNAME( "CvLMNN::train" );

	__BEGIN__;

	int _count, _dims;
	double* db;
	int* lbl;
	int* idx;

	int cls_lbl;
	CvSeq* cls;

	_dims = _train_data->cols;
	_count = _train_data->rows;

	if ( !update_base && updated )
		clear();

	if ( update_base && _dims != var_count )
		CV_ERROR( CV_StsBadArg, "The newly added data have different dimensionality" );

	if ( !update_base || !updated )
	{
		var_count = _dims;
		rebuild_cache();
	}

	db = _train_data->data.db;
	lbl = _responses->data.i;
	idx = _sample_idx->data.i;

	cls_lbl = -1;
	cls = 0;
	for ( int i = 0; i < _count; i++ )
	{
		if ( *lbl != cls_lbl )
		{
			cls_lbl = *lbl;
			cls = cgclass->get( cls_lbl );
			if ( cls == 0 )
			{
				cls = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvVector), cc_storage );
				cvSeqPush( clsidx, &cls_lbl );
				cgclass->add( cls_lbl, cls );
			}
		}
		CvVector vector = cvVector( *idx, db );
		cvSeqPush( cls, &vector );
		db += _dims;
		lbl++;
		idx++;
	}

	CvSparseMatIterator mat_iterator;
	CvSparseNode* node;
	node = cvInitSparseMatIterator( _target_neighbors, &mat_iterator );
	for ( ; node != 0; node = cvGetNextSparseNode( &mat_iterator ) )
	{
		const int* idx = CV_NODE_IDX( _target_neighbors, node );
		cvSetReal2D( target_neighbors, idx[0], idx[1], 1 );
	}
	
	if ( _negatives != NULL )
	{
		node = cvInitSparseMatIterator( _negatives, &mat_iterator );
		for ( ; node != 0; node = cvGetNextSparseNode( &mat_iterator ) )
		{
			const int* idx = CV_NODE_IDX( _negatives, node );
			cvSetReal2D( negatives, idx[0], idx[1], 1 );
		}
		use_negatives = true;
	}

	trainLMNN();

	updated = ok = true;

	__END__;

	return ok;
}

CvMat* CvLMNN::abstract( CvMat* sample, CvMat* response )
{
	CV_FUNCNAME( "CvLMNN::abstract" );

	__BEGIN__;

	if ( response == 0 )
		response = cvCreateMat( params.dims, 1, CV_64FC1 );
	
	cvMatMul( L, sample, response );

	__END__;

	return response;
}

CvMat* CvLMNN::reconstruct( CvMat* sample, CvMat* response )
{
	CV_FUNCNAME( "CvLMNN::reconstruct" );

	CvMat* IL;

	__BEGIN__;

	if ( response == 0 )
		response = cvCreateMat( var_count, 1, CV_64FC1 );

	IL = cvCreateMat( L->cols, L->rows, CV_64FC1 );
	cvInvert( L, IL, CV_SVD );
	cvMatMul( IL, sample, response );
	cvReleaseMat( &IL );

	__END__;

	return response;
}

void CvLMNN::write( CvFileStorage* fs, const char* name )
{
	CV_FUNCNAME( "CvLMNN::write" );

	__BEGIN__;

	cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_ML_LMNN );

	CV_CALL( cvWrite( fs, "L", L ) );

	cvEndWriteStruct( fs );

	__END__;
}

void CvLMNN::read( CvFileStorage* fs, CvFileNode* root_node )
{
	CV_FUNCNAME( "CvLMNN::read" );

	__BEGIN__;

	CV_CALL( L = (CvMat*)cvReadByName( fs, root_node, "L" ) );

	__END__;
}
