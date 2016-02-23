class array_group_sum{
public:
	float *one;
	float alpha;
	float beta;
	int  num;
	int dimen;
	array_group_sum(int dmn,int n);
	~array_group_sum();
	void sum(float *output,float *array_group);
};
class perceptrons{
private:
	array_group_sum *a_sum;
	float *tmp_array;	
	int len;
	void free_mem();
	int block_y;
	float alpha;
	float beta;
	float a_alpha;
public:
	int input_dimen;
	int nodes_num;
	char type;//l:line,t:tanh,s:sigmoid
	int data_num;
	int weight_len; 
	float *weight;
	float *output;	
	float *deriv;
	float decay;
	char decay_mod;
	perceptrons();
	~perceptrons();
	void read_struct(ifstream *fin);
	void write_struct(ofstream *fin);
	void init_struct();
	void init_struct(int dimen,int nodes,char tp);
	void show_struct(int idx);
	int memery();
	void set_data_num(int num);
	void run(float *input);
	void weight_rand();
	void get_deriv(float *input);
	void get_sub_deriv(float *sub_deriv,char sub_type);
	float weight_decay_sum();
	void get_decay_deriv();
};

