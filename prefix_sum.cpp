#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

// Sequential Prefix Sum Function
vector<int> sequentialPrefixSum(const vector<int>& arr) {
    int n = arr.size();
    vector<int> prefixSum(n);
    prefixSum[0] = arr[0];
    for (int i = 1; i < n; ++i) {
        prefixSum[i] = prefixSum[i - 1] + arr[i];
    }
    return prefixSum;
}

vector<int> improvedParallelPrefixSum(const std::vector<int>& arr) {
    int n = arr.size();
    vector<int> prefixSum(arr);
    int numThreads;

    // Store partial sums calculated by each thread
    vector<int> partialSums;

    #pragma omp parallel
    {
        int threadId = omp_get_thread_num();
        numThreads = omp_get_num_threads();
        int chunkSize = (n + numThreads - 1) / numThreads; // Divide array into chunks
        int start = threadId * chunkSize; // calculates start index of the thread's chunk
        int end = min(start + chunkSize, n);

        // Step 1: Compute partial prefix sum for each chunk
        for (int i = start + 1; i < end; ++i) {
            prefixSum[i] += prefixSum[i - 1];
        }

        // Step 2: Store the sum of the current chunk for later use in calculating offsets
        #pragma omp barrier // Synchronize all threads before storing partial sum

        #pragma omp single
        {
            partialSums.resize(numThreads); // this make sures each thread holds one value by resizing the vector
        }

        // saves the last value of the current chunk for future offset calculation
        if (end > start) {
            partialSums[threadId] = prefixSum[end - 1];
        } else {
            partialSums[threadId] = 0;
        }

        #pragma omp barrier // Ensure all partial sums are stored before proceeding

        // Step 3: Calculate offsets based on partial sums from previous threads
        int offset = 0;
        for (int i = 0; i < threadId; ++i) {
            offset += partialSums[i];
        }

        // Step 4: Apply the offset to each element in the chunk
        for (int i = start; i < end; ++i) {
            prefixSum[i] += offset;
        }
    }

    return prefixSum;
}

vector<int> blelloch_scan(vector<int>& x) {
    vector<int> data = x;
    int n = data.size();
    int levels = ceil(log2(n));

    // Upsweep phase
    for (int level = 0; level < levels; ++level) {
        int step = 1 << (level + 1); // calculates step size of current level i.e 2^(level + 1)
        int halfStep = step >> 1; // calculate half of the step size
        
        #pragma omp parallel for
        for (int i = 0; i < n; i += step) {
            if (i + step - 1 < n) {
                data[i + step - 1] += data[i + halfStep - 1];
            }
        }
    }

    // Set last element to 0 for exclusive scan
    int last = data[n - 1];
    if (n > 0) {
        data[n - 1] = 0;
    }

    // Downsweep phase
    for (int level = levels - 1; level >= 0; --level) {
        int step = 1 << (level + 1);
        int halfStep = step >> 1;
        
        #pragma omp parallel for
        for (int i = 0; i < n; i += step) {
            if (i + step - 1 < n) {
                int leftNode = data[i + halfStep - 1];
                data[i + halfStep - 1] = data[i + step - 1];
                data[i + step - 1] += leftNode;
            }
        }
    }

    // Restore the original last element for the final output
    data.push_back(last);
    data.assign(data.begin() + 1, data.end());
    return data;
}

void writeTimesToAFile(double sequentialTime, double parallelTime, double speedup) {
    ofstream outFile("times.txt");
    outFile << sequentialTime << " " << parallelTime << " " << speedup << endl;
    outFile.close();
}

int main() {
    //vector<int> x = {2, 4, 6, 8, 1, 3, 5, 7};

    //remove the comment to test with larger input size
    int largeSize = 1e6; // 1 million elements
	vector<int> x(largeSize);
	for (int i = 0; i < largeSize; ++i) {
	    x[i] = rand() % 100; // Fill with random values between 0 and 99
	}


    // Measure time for Sequential Prefix Sum
    double start = omp_get_wtime();
    vector<int> sequentialResult = sequentialPrefixSum(x);
    double sequentialTime = omp_get_wtime() - start;

    // Print Sequential Result
    cout << "Sequential Prefix Sum: ";
    for (int i : sequentialResult) {
        cout << i << " ";
    }
    cout << "\nTime taken for Sequential Prefix Sum: " << sequentialTime << " seconds" << endl;

    // Measure time for Parallel Prefix Sum
    start = omp_get_wtime();

    //vector<int> parallelResult = blelloch_scan(x); // remove the comment to check it's time
    // because blelloch scan runs slower even with large input, this runs parallelly and is faster for large input
    vector<int> parallelResult = improvedParallelPrefixSum(x);
    double parallelTime = omp_get_wtime() - start;

    // Print Parallel Result with blelloch scan
    cout << "Parallel Prefix Sum: ";
    for (int i : parallelResult) {
        cout << i << " ";
    }
    cout << "\nTime taken for Parallel Prefix Sum: " << parallelTime << " seconds" << endl;

    // Calculate and display speedup
    double speedup = sequentialTime / parallelTime;
    cout << "Speedup (Sequential / Parallel): " << speedup << endl;

   // Write times to a file for visualization in Python
    writeTimesToAFile(sequentialTime, parallelTime, speedup);

    // Call Python script for plotting
    system("python plot_speedup.py");
    return 0;
}
