namespace MachineGlobal{
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
	void checkFold(string path)	{
		struct _stat fl;
		string t = path;
		int i=path.size();
		if(t[i-1]=='\\'){
			t[i-1]=0;
		}
		if (!((_stat(t.c_str(), &fl) == 0) && (fl.st_mode & _S_IFDIR))){
			if(_mkdir(path.c_str()) !=0){
				Assert("无法创建文件夹："+ path);
			}
		}
	}
	int copyFile(string SourceFile,string NewFile){
		ifstream in;
		ofstream out;
		in.open(SourceFile,ios::binary);
		if(in.fail()){
			Assert("无法打开"+ SourceFile);
			in.close();
			out.close();
			return 0;
		}
		out.open(NewFile,ios::binary);
		if(out.fail()){
			Assert("无法创建"+ NewFile);
			out.close();
			in.close();
			return 0;
		}
		else{
			out<<in.rdbuf();
			out.close();
			in.close();
			return 1;
		}
	}
	const static int NullValid = -1;
	
	const static int MainFile = -1;
	const static int AllMachine = -2;
	const static int TrainDataSet = 1;
	const static int ValidDataSet = 2;
	const static int TestDataSet = 3;
}
using namespace MachineGlobal;
template <typename TYPE, bool CUDA>
class dataSetBase;
template<typename TYPE, bool CUDA>
class seriesDataBase;
template<typename TYPE, bool CUDA>
class MachineBase;