template <bool CUDA>
class sinSample: public dataSetBase<float, CUDA>{
private:
	int samplesNum;	
public:
	virtual void getDataSet(){		
		float * tmpX = new float[samplesNum];
		float * tmpT = new float[samplesNum];
		for(int i = 0; i < samplesNum; i++){
			float t = 3.14 * float(rand()%10000 -5000)/10000;
			tmpX[i] = t;
			tmpT[i] = sin(t);
			
		}
		//以上代码生成数据

		loadDataSet(tmpX, tmpT,  samplesNum);

		delete [] tmpX;
		delete [] tmpT;

	}
	sinSample(int num){		
		init(1, 1);	
		samplesNum = num;		
	}
};

#include "sampleDataSet/mnist.h"
template <bool CUDA>
class mnistSample: public dataSetBase<float, CUDA>, mnist{
public:
	mnistSample(string path):mnist(path){
		init(input_num, output_num);
		set("activeFunc", SIGMOID);		
	}
	void createSamples(){	
		loadSamples(data, label_d, train_num);
	}
	void showResult(){	
		float * Yd = Yv[0].T().getData();
		float * Td = Tv[0].T().getData();
		accuracy(Yd, Td, validNum);
		delete [] Yd;
		delete [] Td;
	}
	virtual void showValidsResult(MatGroup<float, CUDA> &T, MatGroup<float, CUDA> &Y){
		float ret = 0;
		int num = 0;
		cout<<"\n";
		for(int i = 0 ; i < T.num(); i++){
			float * Td = (T[i].T()).getData();
			float * Yd = (Y[i].T()).getData();
			ret += accuracy(Yd, Td, T[i].rows()) * T[i].rows();
			num +=   T[i].rows();
			cout<<"\n";
			delete [] Yd;
			delete [] Td;
			getchar();
		}
		cout<<"\naccuracy"<<(ret/num);
	};
	virtual void pauseAction(MachineBase<float, CUDA> * _this){
		_this->save(true);
	}
};
template <typename TYPE, bool CUDA>
class singleSeries: public seriesDataBase<TYPE, CUDA>{
protected:
	string path;
	
public:
	void getDataSet(){
		if(!fileIsExist(path)){
			Assert("\n文件不存在:" + path);
		}
		ifstream fl(path);
		string line;
		vector<TYPE> ret;
		while (getline (fl, line)){
			ret.push_back((TYPE)atof(line.c_str()));
		}
		fl.close();
		int allLen = ret.size() - 1;
		loadDataSet(ret.data(), ret.data() + 1, 1, ret.size() - 1);		
	}
	singleSeries(string _path):seriesDataBase(){		
		init(1, 1);		
		path = _path;		
	}
};
template <typename TYPE, bool CUDA>
struct stockIndex:public seriesDataBase<TYPE, CUDA>{
private:
	string path;
	void createSamples(){//返回dtNum
		int all_len;
		int input_dimen;
		ifstream fl(path, ios::binary);
		fl.read((char *) & all_len, sizeof(int));
		fl.read((char *) & input_dimen, sizeof(int));
		TYPE * tmp = new TYPE[all_len * input_dimen];
		fl.read((char *) tmp, _msize(tmp));
		fl.close();
		MatCF x(input_dimen, all_len);
		x.importData(tmp);
		MatCF t = x.row(7);
		x = x.removeRightCols();
		t = t.removeLeftCols();
		all_len --;
		TYPE * srcX = new TYPE[all_len * inputNum];
		TYPE * srcT = new TYPE[all_len * outputNum];
		x.exportData(srcX);
		t.exportData(srcT);
		loadSamples(srcX, srcT, 1, all_len);
		delete [] srcX;
		delete [] srcT;
	}	
public:
	stockIndex( string _path):seriesDataBase(){
		path = _path;
		ifstream fl(path, ios::binary);
		if( !fl){
			Assert("\n文件不存在:" + path);
		}
		int input_num;
		fl.read((char *) & input_num, sizeof(int));
		fl.read((char *) & input_num, sizeof(int));
		fl.close();
		init(input_num, 1);
		set("activeFunc",LINEAR);
		maxCorr = -2;
	}	
	void showResult(){
		showValidCorrel();
	}
	void pauseAction(){
		string s;
		cout<<"输入s保存csv";
		cin>>s;
		if(s[0] == 's'){
			outputValidCsv("f:\\stockIdxcsv.csv");
		}
	}
	virtual void showValidsResult(MatGroup<TYPE, CUDA> &T, MatGroup<TYPE, CUDA> &Y){
		cout<<"\ncorr:"<<T.correl(Y);
	};
};
