template <typename TYPE, bool CUDA>
class  matMem{
	friend class matCore<TYPE, CUDA>;
public:
	TYPE * mem;
	eigenMat *eigenMem; 
	int count;
	matMem();
	matMem(int rows, int cols);
	~ matMem();	
};