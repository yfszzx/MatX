template <typename TYPE, bool CUDA>
class  matMem{
	friend class matCore<TYPE, CUDA>;
public:
	TYPE * mem;
	eigenMat *eigenMem; 
	int count;
	bool fixMem;
	matMem();
	matMem(TYPE * src);
	matMem(int rows, int cols);
	~ matMem();	
};