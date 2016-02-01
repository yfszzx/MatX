namespace matrixGlobal{
	void Global_debug(int line, const char *file){
		cout << "\ndebug at line:" << line << "  file:" << file <<endl;
		getchar();
	}
	void Global_Assert(string illu, int line, const char *file){
		cout << "\n¡¾´íÎó¡¿" << illu;
		cout << "\nat line:" << line << "  file:" << file <<endl;
		string s;
		cin >> s;
	}

	namespace cpu_funcs{
		template <typename TYPE>
		TYPE sigm(TYPE x){
			return 1.0f/(1.0f+expf(-x));
		}
		template <typename TYPE>
		TYPE square(TYPE x){
			return x*x;
		}
		template <typename TYPE>
		TYPE tanh(TYPE x){
			return 1.0f-2.0f/(1.0f+ expf(2*x));
		}
		template <typename TYPE>
		TYPE abs(TYPE x){			
			return (x>0)?x:(-x);
		}
		template <typename TYPE>
		TYPE Gre(TYPE x, TYPE y){
			return TYPE(x > y);
		}
		template <typename TYPE>
		TYPE GreEqu(TYPE x, TYPE y){
			return TYPE(x >= y);
		}
		template <typename TYPE>
		TYPE Les(TYPE x, TYPE y){
			return TYPE(x < y);
		}
		template <typename TYPE>
		TYPE LesEqu(TYPE x, TYPE y){
			return TYPE(x <= y);
		}
		template <typename TYPE>
		TYPE Equ(TYPE x, TYPE y){
			return TYPE(x == y);
		}
		template <typename TYPE>
		TYPE NotEqu(TYPE x, TYPE y){
			return TYPE(x != y);
		}

	}
	const bool randDebug = false;
	void MatXInit(bool showMatXInfo = true){
		if(showMatXInfo){
			if(MatX_USE_GPU){
				cout<<"\nMOD: GPU/CPU";
			}else{
				cout<<"\nMOD: CPU ONLY";
			}
			if(randDebug){
				cout<<"\nRand Debug mod";
			}
		}
		cuWrap::initCuda(showMatXInfo);
		if(randDebug){
			srand(0);			
		}else{
			srand(time(NULL));
		}
		if(showMatXInfo){
			cout<<"\nMatX started\n";
		}
	}
};
using namespace matrixGlobal;
template <typename TYPE, bool CUDA>
class matCore;
template <typename TYPE, bool CUDA>
class MatriX;
template <typename TYPE, bool CUDA>
class MatGroup;

typedef MatriX<float, false> MatCF;
typedef MatriX<float, MatX_USE_GPU> MatGF;
typedef MatriX<double, false> MatCD;
typedef MatriX<double, MatX_USE_GPU> MatGD;
typedef MatGroup<float, false> MatCFG;
typedef MatGroup<float, MatX_USE_GPU> MatGFG;
typedef MatGroup<double, false> MatCDG;
typedef MatGroup<double, MatX_USE_GPU> MatGDG;
typedef MatGroup<double, false> MatGrpCD;
typedef MatGroup<double, MatX_USE_GPU> MatGDG;
typedef Matrix<float, -1, -1> eMatF;
typedef Matrix<double, -1, -1> eMatD;

#define MatX MatriX<TYPE, CUDA>
#define MatXG MatGroup<TYPE, CUDA>
#define MatXN MatriX<TYPE, !CUDA>
#define MatXNG MatGroup<TYPE, !CUDA>
#define eigenMat Matrix<TYPE, -1, -1>

#define Assert(illu) Global_Assert((illu), __LINE__, __FILE__);
#define Debug(x) cout<<endl<<(x);Global_debug( __LINE__, __FILE__);
