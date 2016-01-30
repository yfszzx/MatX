/*
安装 glut  
GLUT3.7下载地址：http://www.opengl.org/resources/libraries/glut/glutdlls37beta.zip
点击上面的链接下载最新的GLUT，最新的GLUT版本是3.7，解压，将 glut32.dll 和 glut.dll 拷贝到  c:\windows\system32 下面，将 glut32.lib 和 glut.lib 拷贝到 VC 安装目录下的 lib 目录下（如：\Microsoft Visual Studio 9.0\VC\lib\下），将 glut.h 拷贝到VC安装目录下的 \include\gl\ 目录下（如：\Microsoft Visual Studio 9.0\VC\include\gl\下）
*/
#include<gl/glut.h>
namespace Plot{
	float *X;
	float *Y;
	int num;
	float scaleX;
	float scaleY;
	float biasX;
	float biasY;
	string title = "default";
	const int high = 400;
	const int width = 400;

	void initPlot(int* argc,char** argv){
		 glutInit(argc, argv);  
		 glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
         glutInitWindowPosition(width/2, high/2);
         glutInitWindowSize(width, high);
	}
	void draw(){
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
         glLoadIdentity();
         glBegin(GL_POINTS);
		 for(int i = 0; i<num; i++){
			  glVertex3f( X[i] * scaleX + biasX, Y[i] * scaleY + biasY, 0);
		 }
         glEnd();
         glutSwapBuffers();
	}
	void plot(float * _X, float * _Y, int _num){
		X = _X;
		Y = _Y;
		num = _num;
		float Xmax = *max_element(X, X + num);
		float Xmin = *min_element(X, X + num);
		float Ymax = *max_element(Y, Y + num);
		float Ymin = *min_element(Y, Y + num);
		scaleX = float(width)/(Xmax - Xmin);
		scaleY = float(width)/(Ymax - Ymin);
		biasX = -(Xmax + Xmin)/2 * scaleX;
		biasY = -(Ymax + Ymin)/2 * scaleY;
		glutCreateWindow(title.c_str());
        glutDisplayFunc(draw);
        glutMainLoop();

	}
}