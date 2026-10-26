#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>

// Per-process kernel counter files written by the CUPTI injection library.
// Each process writes kernels_count.<pid> so workers cannot overwrite each other.
// DEPO sums those files. A missing directory or a half-written file contributes nothing.

inline bool isDepoKernelCounterFile(const std::string& name)
{
    if (name == "kernels_count" || name.rfind("kernels_count.", 0) == 0)
    {
        return true;
    }
    return name.rfind("kernels_gpu_", 0) == 0;
}

inline unsigned long long sumDepoKernelCountFiles()
{
    namespace fs = std::filesystem;
    unsigned long long sum = 0;
    std::error_code ec;
    fs::directory_iterator it(".", ec);
    if (ec)
    {
        return 0;
    }
    const fs::directory_iterator end;
    for (; it != end; it.increment(ec))
    {
        if (ec)
        {
            break;
        }
        if (!it->is_regular_file())
        {
            continue;
        }
        const std::string name = it->path().filename().string();
        if (!(name == "kernels_count" || name.rfind("kernels_count.", 0) == 0))
        {
            continue;
        }
        std::ifstream in(it->path());
        if (!in)
        {
            continue;
        }
        std::string line;
        if (!std::getline(in, line) || line.empty())
        {
            continue;
        }
        try
        {
            const long long value = std::stoll(line);
            if (value > 0)
            {
                sum += static_cast<unsigned long long>(value);
            }
        }
        catch (const std::exception&)
        {
        }
    }
    return sum;
}

inline void clearDepoKernelCounterFiles()
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::directory_iterator it(".", ec);
    if (ec)
    {
        return;
    }
    const fs::directory_iterator end;
    for (; it != end; it.increment(ec))
    {
        if (ec)
        {
            break;
        }
        if (!it->is_regular_file())
        {
            continue;
        }
        if (isDepoKernelCounterFile(it->path().filename().string()))
        {
            fs::remove(it->path(), ec);
        }
    }
}
