#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>

struct Student {
    std::string name;
    double gpa;
};

// ------------------- Counting Sort (Stable) -------------------
void countingSort(std::vector<Student>& arr) {
    int n = arr.size();
    if (n == 0) return;

    std::vector<int> gpas(n);
    for (int i = 0; i < n; i++)
        gpas[i] = static_cast<int>(arr[i].gpa * 100);

    int maxGPA = *std::max_element(gpas.begin(), gpas.end());

    std::vector<int> count(maxGPA + 1, 0);
    for (int i = 0; i < n; i++)
        count[gpas[i]]++;

    for (int i = 1; i <= maxGPA; i++)
        count[i] += count[i - 1];

    std::vector<Student> output(n);
    for (int i = n - 1; i >= 0; i--) {
        output[count[gpas[i]] - 1] = arr[i];
        count[gpas[i]]--;
    }

    arr = output;
}

// ------------------- Selection Sort (Unstable) -------------------
void selectionSort(std::vector<Student>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++)
            if (arr[j].gpa < arr[min_idx].gpa)
                min_idx = j;
        std::swap(arr[i], arr[min_idx]);
    }
}

// ------------------- Read and Write -------------------
std::vector<Student> readData(const std::string& filename) {
    std::vector<Student> data;
    std::ifstream infile(filename);
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        Student s;
        if (!(iss >> s.name >> s.gpa)) break;
        data.push_back(s);
    }
    return data;
}

void writeData(const std::vector<Student>& arr, const std::string& filename) {
    std::ofstream outfile(filename);
    for (const auto& s : arr)
        outfile << s.name << " " << std::fixed << std::setprecision(2) << s.gpa << "\n";
}

// ------------------- Display Sample -------------------
void displaySample(const std::vector<Student>& arr, int sampleSize = 10) {
    for (int i = 0; i < std::min(sampleSize, (int)arr.size()); i++)
        std::cout << arr[i].name << " " << std::fixed << std::setprecision(2) << arr[i].gpa << std::endl;
}

// ------------------- Main -------------------
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./sorting [10k|100k|1M]\n";
        return 1;
    }

    std::string arg = argv[1];
    std::string filename;

    if (arg == "10k") filename = "students_10000.txt";
    else if (arg == "100k") filename = "students_100000.txt";
    else if (arg == "1M") filename = "students_1000000.txt";
    else {
        std::cerr << "Invalid argument. Use 10k, 100k, or 1M.\n";
        return 1;
    }

    std::vector<Student> data = readData(filename);
    std::cout << "Loaded " << data.size() << " records from " << filename << "\n\n";

    // Counting Sort
    auto stableData = data;
    auto start = std::chrono::high_resolution_clock::now();
    countingSort(stableData);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Counting Sort (Stable) Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";
    writeData(stableData, "counting_sort.txt");
    std::cout << "Counting Sort result exported to counting_sort.txt\n\n";

    // Selection Sort (Unstable) on full dataset
    auto unstableData = data;
    start = std::chrono::high_resolution_clock::now();
    selectionSort(unstableData);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Selection Sort (Unstable) Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";
    writeData(unstableData, "selection_sort.txt");
    std::cout << "Selection Sort result exported to selection_sort.txt\n";

    return 0;
}
