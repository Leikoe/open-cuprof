#include <iostream>
#include <fstream>
#include <cstdint>

int main() {
    std::ifstream in("trace.json");
    std::string line;
    int count = 0;
    while (std::getline(in, line) && count < 20) {
        std::cout << line << std::endl;
        count++;
    }
    return 0;
}
