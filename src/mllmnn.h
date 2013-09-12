#ifndef GUARD_mllmnn_h
#define GUARD_mllmnn_h

#include <ml.h>
#include <map>

class CvCategoricalClasses;

#define CV_TYPE_NAME_ML_LMNN      "opencv-ml-large-margin-nearest-neighbor"

struct CV_EXPORTS CvLMNNParams
{
	double stepsize;
	double minstepsize;
	double u;
	double margin;
	double marginblur;
	double epsilon;
	int maxconverge;
	int maxiter;
	int maxidx;
	int reinitialize;
	int dims;
	CvLMNNParams( int _dims )
	: stepsize(1e-4), minstepsize(1e-9), u(.5), margin(1), marginblur(1), epsilon(1e-4), maxconverge(10), maxiter(100), maxidx(1000000), reinitialize(10), dims(_dims)
	{}
	CvLMNNParams(double _stepsize, double _minstepsize, double _u, double _margin, double _marginblur, double _epsilon, int _maxcoverage, int _maxiter, int _maxidx, int _reinitialize, int _dims )
	: stepsize(_stepsize), minstepsize(_minstepsize), u(_u), margin(_margin), marginblur(_marginblur), epsilon(_epsilon), maxconverge(_maxcoverage), maxiter(_maxiter), maxidx(_maxidx), reinitialize(_reinitialize), dims(_dims)
	{}
};

class CV_EXPORTS CvLMNN : public CvStatModel
{
	private:
		const CvLMNNParams params;
		CvMemStorage* storage;
		CvCategoricalClasses* cgclass;
		CvSparseMat* target_neighbors;
		CvSparseMat* negatives;
		CvSeq* clsidx;
		int var_count;
		int total;
		bool updated;
		bool rebuilt;
		bool use_negatives;
		void trainLMNN();

		CvMat *C, *Cl, *Cij, *G, *dL, *tL, *L, *Lp;
		CvSparseMat* N;
		CvMemStorage *cc_storage, *active_set_storage, *active_set_child_storage;
		CvSeq* active_set;
		double norm( const CvMat* arr );
		void rebuild_cache();
		void rebuild_active_set();
		bool expand_active_set();
	public:
		CvLMNN( const CvLMNNParams _params );
		~CvLMNN();
		
		virtual void clear();
		virtual void write( CvFileStorage* fs, const char* name );
		virtual void read( CvFileStorage* fs, CvFileNode* root_node );
		virtual bool train( const CvMat* _train_data, const CvMat* _responses, const CvMat* _sample_idx, const CvSparseMat* _target_neighbors, const CvSparseMat* _negatives = 0, bool update_base = 0 );
		CvMat* abstract( CvMat* sample, CvMat* response = 0 );
		CvMat* reconstruct( CvMat* sample, CvMat* response = 0 );
};

#endif
