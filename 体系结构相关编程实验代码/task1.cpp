#include <iostream>
#include <windows.h>
#include <stdlib.h>
using namespace std;
const int N = 1000;
int count = 500;
double b[N][N], a[N],sum[N];
void init()
{
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            b[i][j] = i + j;
            a[j] = j;
        }
    }
}
int main()
{
    long long head, tail, freq; // timers
    double total_time = 0.0;
    init();
    /*普通算法
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        //start time
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        for(int i = 0; i<N; i++)
        {
            sum[i] = 0.0;
            for(int j = 0; j < N; j++)
            {
                sum[i] += a[i] * b[j][i];
            }
        }

        //end time
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        total_time += (tail - head) * 1000.0 / freq;




    */

    /*cache算法
            QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
            //start time
            QueryPerformanceCounter((LARGE_INTEGER*)&head);

            for(int i = 0; i<N; i++)
            {
                sum[i] = 0.0;
            }

            for(int k = 0;k<N;k++)
            {
                for(int i = 0;i<N;i++)
                {
                    sum[i] += b[k][i]*a[k];
                }
            }
            //end time
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            total_time += (tail - head) * 1000.0 / freq;
    */

    for(int j = 0; j<count; j++)
    {
        //循环展开
        double tem1 = 0.0, tem2 = 0.0;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        //start time
        QueryPerformanceCounter((LARGE_INTEGER*)&head);

        for(int i = 0; i<N; i+=2)
        {
            sum[i] = 0.0;
            sum[i+1] = 0.0;
            tem1 = 0.0;
            tem2 = 0.0;
            for(int j = 0; j < N; j+=2)
            {
                sum[i] += a[i] * b[j][i];
                tem1 += a[i] * b[j+1][i];

                sum[i+1] += a[i+1] * b[j][i+1];
                tem2 += a[i+1]*b[j+1][i+1];
            }
            sum[i] += tem1;
            sum[i+1] += tem2;
        }
        //end time
        QueryPerformanceCounter((LARGE_INTEGER*)&tail);
        total_time += (tail - head) * 1000.0 / freq;
    }
    cout <<"N:"<<N<<endl;
    cout <<"Count:" << count << endl;
    cout << "Total Time: " << total_time <<"ms" << endl;
    cout << total_time / count;

    return 0;
}

