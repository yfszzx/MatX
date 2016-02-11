namespace ANNGlobal{
	class timer{
		double startTime;
	public:
		clock_t start;
		clock_t stop;
		void show(bool reset = false){
			cout<<"\nUse Time:"<<get(reset);
		}
		void set(double tm = 0){
			start = clock();
			startTime = tm;
		}
		double get(bool reset = false){
			stop = clock();
			double t = double(stop-start)/CLOCKS_PER_SEC + startTime;
			if ( reset)set();
			return t;
		}
		timer(){
			set();
		}


	};
#define Mrand(x) ((rand()%(RAND_MAX+1)*(RAND_MAX+1)+rand())%(x))
	int randSeeder;
	enum activeFunctionType{ LINEAR = 0, TANH = 1, SIGMOID = 2};
	template <typename TYPE, bool CUDA>
	inline MatriX<TYPE, CUDA> activeFunc(int func, const MatriX<TYPE, CUDA> & X){
		switch(func){
		case LINEAR:
			return X;
		case TANH:
			return  tanh(X);
		case SIGMOID:
			return  sigm(X);
		}
	}
	template <typename TYPE, bool CUDA>
	inline MatriX<TYPE, CUDA> activeDerivFunc(int func, const MatriX<TYPE, CUDA> & diff, const MatriX<TYPE, CUDA> & Y){
		switch(func){
		case LINEAR:
			return diff;
		case TANH:
			return  diff.cwiseProduct ( (TYPE)1.0f - square(Y));	
		case SIGMOID:
			return diff.cwiseProduct (Y.cwiseProduct ( (TYPE)1.0f - Y));	
		}
	}
	bool fileIsExist(string path){
		bool ret = true;
		ifstream fl(path);
		if(!fl){
			ret =false;
		}
		fl.close();
		return ret;
	}


}
using namespace ANNGlobal;
template <typename TYPE, bool CUDA>
class dataSetBase;
template<typename TYPE, bool CUDA>
class seriesDataBase;
template<typename TYPE, bool CUDA>
class MachineBase;