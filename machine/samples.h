template <typename TYPE, bool CUDA>
class sinSample: public dataSetBase<TYPE, CUDA>{
public:
	sinSample(int num, int crossNum, bool rand){
		init(1, 1, LINEAR, rand, crossNum);
		load(num);
	}

	void loadData(){		
		TYPE *tX = new TYPE[dataNum];
		TYPE *tT = new TYPE[dataNum];
		for(int i = 0; i<dataNum; i++){
			tX[i] = 3.1415 * TYPE(Mrand(10000) -5000)/10000;
			tT[i] = sin(tX[i]);
		}
		X0[0].importData(tX);
		T0[0].importData(tT);
	}

	void showResult(){
		MatGroup<TYPE, CUDA> cY(Yv, seriesLen);
		MatGroup<TYPE, CUDA> cT(Tv, seriesLen);
		cout<<"\ncorrel:"<<cY.correl(cT);
	}
	virtual void showValidsResult(MatGroup<TYPE, CUDA> &T, MatGroup<TYPE, CUDA> &Y){
		cout<<"\ncorr:"<<T.correl(Y);
	};

};

#include "sampleDataSet/mnist.h"
template <bool CUDA>
class mnistSample: public dataSetBase<float, CUDA>, mnist{
public:
	mnistSample(bool rand, int crossNum, string path):mnist(path){
		init(input_num, output_num, SIGMOID, rand, crossNum);
		load(train_num);
	}
	void loadData(){		
		X0[0].importData(data);
		T0[0].importData(label_d);		
	}
	void showResult(){	
		float * Yd;
		float * Td;
		Tv[0].transpose().exportData(Td);
		Yv[0].transpose().exportData(Yd);
		accuracy(Yd, Td, validNum);
		delete [] Yd;
		delete [] Td;
	}
	virtual void showValidsResult(MatGroup<float, CUDA> &T, MatGroup<float, CUDA> &Y){
		float ret = 0;
		int num = 0;
		for(int i = 0 ; i < T.num(); i++){
			float * Yd;
			float * Td;
			T[i].T().exportData(Td);
			Y[i].T().exportData(Yd);
			ret += accuracy(Yd, Td, T[i].rows()) * T[i].rows();
			num +=   T[i].rows();
			cout<<"\n";
			delete [] Yd;
			delete [] Td;
		}
		cout<<"\n"<<T.num();
		cout<<"\naccuracy"<<(ret/num);
	};
};
template <typename TYPE, bool CUDA>
class singleSeries: public seriesDataBase<TYPE, CUDA>{
private:
	string path;
	//	MatGroup fixT;
	//MatGroup fixY;
	int currentValidIdx;
	int loadSeries(){//返回dtNum
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
		int allLen = ret.size()-1;
		srcX = new TYPE[allLen];
		srcT = new TYPE[allLen];
		memcpy(srcX, ret.data(), _msize(srcX));
		memcpy(srcT, ret.data() + 1, _msize(srcT));
		int dtNum = (allLen - preLen)/(seriesLen - preLen);
		cutSeries(dtNum);
		return dtNum;
	}
public:
	singleSeries(bool rand, int _seriLen, int _preLen, int _crossNum,  string _path):seriesDataBase(_preLen){
		init(1, 1, LINEAR, rand, _crossNum, _seriLen);
		path = _path;
		load();
		currentValidIdx = -1;
	}	
	void showResult(){
		showValidCorrel();
	}
	void pauseAction(){
		string s;
		cout<<"输入s保存csv";
		cin>>s;
		if(s[0] == 's'){
			outputValidCsv("f:\\signlecsv.csv");
		}
	}
	virtual void showValidsResult(MatGroup<TYPE, CUDA> &T, MatGroup<TYPE, CUDA> &Y){
		cout<<"\ncorr:"<<T.correl(Y);
	};

};
template <typename TYPE, bool CUDA>
struct stockIndex:public seriesDataBase<TYPE, CUDA>{
private:
	string path;
	int loadSeries(){//返回dtNum
		ifstream fl(path, ios::binary);
		int all_len;
		int input_dimen;
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
		srcX = new TYPE[all_len * inputNum];
		srcT= new TYPE[all_len * outputNum];
		x.exportData(srcX);
		t.exportData(srcT);
		int dtNum = (all_len - preLen)/(seriesLen - preLen);
		cutSeries(dtNum);
		return dtNum;
	}	
public:
	stockIndex(bool rand, int _seriLen, int _preLen, int _crossNum,  string _path):seriesDataBase(_preLen){
		path = _path;
		ifstream fl(path, ios::binary);
		if( !fl){
			Assert("\n文件不存在:" + path);
		}
		int input_num;
		fl.read((char *) & input_num, sizeof(int));
		fl.read((char *) & input_num, sizeof(int));
		fl.close();
		init(input_num, 1, LINEAR, rand, _crossNum, _seriLen);
		load();
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
