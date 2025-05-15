#include <iostream>
#include <queue>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <random>
#include <fstream>
#include <string>
#include <variant>
#include <functional>

#define SIN 0
#define SQRT 1
#define POW 2

class TaskServer {
public:
    using ResultType = double;
    using TaskFunc = std::function<ResultType()>;

    struct Task {
        size_t id;
        TaskFunc func;
        int type;
        double arg1;
        double arg2;
    };

    struct Result {
        ResultType value;
        int type;
        double arg1;
        double arg2;
    };

    TaskServer() : next_id(1), running(false) {}

    void start() {
        running = true;
        server_thread = std::thread(&TaskServer::process_tasks, this);

        std::cout << "server start\n";
    }

    void stop() {
        running = false;
        cv.notify_one();
        if (server_thread.joinable()) {
            server_thread.join();
        }

        std::cout << "server stop\n";
    }

    size_t add_task(int type, double arg1, double arg2 = 0.0) {
        std::lock_guard<std::mutex> lock(mtx);
        size_t id = next_id++;
        
        TaskFunc func;
        switch (type) {
            case SIN: func = [arg1]() { return sin(arg1); }; break;
            case SQRT: func = [arg1]() { return sqrt(arg1); }; break;
            case POW: func = [arg1, arg2]() { return pow(arg1, arg2); }; break;
        }
        
        tasks.push({id, func, type, arg1, arg2});
        cv.notify_one();
        return id;
    }

    Result get_result(size_t id) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_result.wait(lock, [this, id]() { return results.count(id); });
        Result res = results[id];
        results.erase(id);
        return res;
    }

private:
    std::queue<Task> tasks;
    std::unordered_map<size_t, Result> results;
    std::mutex mtx;
    std::condition_variable cv;
    std::condition_variable cv_result;
    std::thread server_thread;
    size_t next_id;
    bool running;

    void process_tasks() {
        while (running) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this]() { return !tasks.empty() || !running; });
                if (!running && tasks.empty()) 
                    break;
                if (tasks.empty()) 
                    continue;
                
                task = std::move(tasks.front());
                tasks.pop();
            }

            ResultType result = task.func();

            {
                std::lock_guard<std::mutex> lock(mtx);
                results[task.id] = {result, task.type, task.arg1, task.arg2};
            }
            cv_result.notify_all();
        }
    }
};


class TaskClient {
public:
    TaskClient(TaskServer& server, int type, const std::string& filename)
        : server(server), type(type), filename(filename) {}

    void run(int count) {
        std::ofstream file(filename);
        file << "ID,Arg1,Arg2,Result\n";
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.1, 10.0);

        for (int i = 0; i < count; ++i) {
            double arg1 = dis(gen);
            double arg2 = (type == POW) ? dis(gen) : 0.0;
            
            size_t id = server.add_task(type, arg1, arg2);
            auto result = server.get_result(id);
            
            file << id << ","
                 << result.arg1 << ","
                 << result.arg2 << ","
                 << result.value << "\n";
        }
    }

private:
    TaskServer& server;
    int type;
    std::string filename;
};

int main() {
    TaskServer server;

    auto server_t_start = std::chrono::steady_clock::now();
    server.start();

    const int N = 10000;
    
    auto clients_t_start = std::chrono::steady_clock::now();

    TaskClient sin_client(server, SIN, "sin_results.csv");
    TaskClient sqrt_client(server, SQRT, "sqrt_results.csv");
    TaskClient pow_client(server, POW, "pow_results.csv");

    std::thread t1([&sin_client]() { sin_client.run(N); });
    std::thread t2([&sqrt_client]() { sqrt_client.run(N); });
    std::thread t3([&pow_client]() { pow_client.run(N); });

    t1.join();
    t2.join();
    t3.join();

    auto clients_t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> clients_time = clients_t_end - clients_t_start;

    server.stop();
    auto server_t_end= std::chrono::steady_clock::now();

    std::chrono::duration<double> server_time = server_t_end - server_t_start;

    std::cout << "Server time: " << server_time.count()
         << "\nClient time: " << clients_time.count() << std::endl;

    return 0;
}