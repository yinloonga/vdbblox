#pragma once

#include <chrono>
#include <cstdlib>
#include <ctime>

namespace vdbblox_tools {
class TicToc {
   public:
    TicToc() { tic(); }

    void tic() { start = std::chrono::system_clock::now(); }

    double toc() {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;  // returns millisec
    }

   private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};
}  // namespace tools
