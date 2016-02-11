class mnist{
private:
	float **label;
	float **label_t;
	float **input;
	float **input_t;
	int rows;
	int columns;
	void showImg(int idx,char type,bool label_show=true);
	int input_idx(char type);
	int value(float *tgt);
	int value(int idx,char mod);
	int get_out_value(float *out);
	void show_out(float *out);
	void show_wrong(float *out,char mod='c');
public:
	float *data;
	float *label_d;
	float *data_t;
	float *label_t_d;	
	int input_num;
	int output_num;	
	int train_num;
	int test_num;
	mnist(string path);	
	void check();	
	float accuracy(float *out,float *tgt,int num);
	
};