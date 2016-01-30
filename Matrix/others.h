template <typename TYPE, bool CUDA>
ostream & operator <<(ostream &os, const MatriX<TYPE, CUDA> &m){
	eigenMat s(m.rows(), m.cols());
	MatriX<TYPE,CUDA> tmp(m, MatriX<TYPE,CUDA>::ALL);
	if(CUDA){		
		cuWrap::memD2H(s.data(), tmp.dataPrt(), sizeof(TYPE) * m.size());
	}else{
		memcpy(s.data(), tmp.dataPrt(), sizeof(TYPE) * m.size());
	}
	os<<"\n"<<s<<"\n";
	return os;
}

template <typename TYPE, bool CUDA>
void MatriX<TYPE, CUDA>::exportData(TYPE * & dt, bool cuda) const{
	MatriX<TYPE, false> tmp = *this;
	tmp.copyRealise(true, true);
	if(cuda){
		cuWrap::malloc((void **)& dt, sizeof(TYPE) * size());
		cuWrap::memH2D(dt, tmp.dataPrt(),  sizeof(TYPE) * size());
	}else{
		dt = new TYPE[size()];
		memcpy(dt, tmp.dataPrt(), _msize(dt));
	}
}
template <typename TYPE, bool CUDA>
void MatriX<TYPE, CUDA>::save(ofstream & fl) const{
	MatriX<TYPE, CUDA> tmpcpu(*this, ALL);
	 MatriX<TYPE, false> tmp = tmpcpu;
	 int c = tmp.cols();
	 int r = tmp.rows();
	 fl.write((char *)&c, sizeof(int));
	 fl.write((char *)&r, sizeof(int));
	 fl.write((char *)tmp.dataPrt(), sizeof(TYPE) * size());
};
template <typename TYPE, bool CUDA>
void MatriX<TYPE, CUDA>::read(ifstream & fl){
	int r;
	int c;
	fl.read((char *)&c, sizeof(int));
	fl.read((char *)&r, sizeof(int));
	MatriX<TYPE, false> tmp(r, c);
	fl.read((char *)tmp.dataPrt(), sizeof(TYPE) * tmp.size());
	*this = tmp;
};

