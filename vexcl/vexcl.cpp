#include <iostream>
#include <vector>
#include <vexcl/vexcl.hpp>

#include "variants.h"

#define N 1000000000
#define RUNS 10

int main(int argc, char** argv) {
    // Parse command line args
    Variant variant = openmp;
    for(int i = 1; i < argc; i++)
    {
        char* arg = argv[i];
        if(strcmp(arg, "--openmp") == 0) {
            variant = openmp;
        } else if(strcmp(arg, "--pthreads") == 0) {
            variant = pthreads;
        } else if(strcmp(arg, "--cthreads") == 0) {
            variant = cthreads;
        } else if(strcmp(arg, "--cuda") == 0) {
            variant = cuda;
        } else if(strcmp(arg, "--async") == 0) {
            variant = async;
        }
    }

    try {
        // Init VexCL context: grab all devices.
        vex::Context ctx(vex::Filter::Any);

        if (ctx.empty()) throw std::runtime_error("No devices found");

        std::cout << "Devices: " << ctx << std::endl;

        // Prepare input data.
        std::vector<double> a(N, 1);
        std::vector<double> b(N, 2);
        std::vector<double> c(N);

        // Allocate device vectors and transfer input data to device.
        vex::vector<double> A(ctx.queue(), a);
        vex::vector<double> B(ctx.queue(), b);
        vex::vector<double> C(ctx.queue(), N);

        // Launch kernel on compute device.
        C = A + B;

        // Get result back to host.
        copy(C, c);

        // Should get '3' here.
        std::cout << c[42] << std::endl;
        return 0;
    } catch (const cl::Error &err) {
        std::cerr << "OpenCL error: " << err << std::endl;
    } catch (const std::exception &err) {
        std::cerr << "Error: " << err.what() << std::endl;
    }
    return 1;
}