/*
   Copyright 2022-2024, Adam Krzywaniak.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#pragma once

#include "data_structures/power_and_perf_result.hpp"

static inline
std::string logCurrentResultLine(
    PowAndPerfResult& curr,
    PowAndPerfResult& first,
    double k,
    bool noNewLine = false)
{
    std::stringstream sstream;
    if (curr.appliedPowerCapInWatts_ < 0.0) {
        sstream << "refer.\t";
    } else {
        sstream << curr.appliedPowerCapInWatts_ << "\t";
    }
    sstream << std::fixed << std::setprecision(2)
              << curr.energyInJoules_ << "\t"
              << curr.averageCorePowerInWatts_ << "\t"
              << curr.filteredPowerOfLimitedDomainInWatts_ << "\t"
              << std::setprecision(3)
              << curr.getInstrPerSecond() / first.getInstrPerSecond() << "\t"
              << curr.getEnergyPerInstr() / first.getEnergyPerInstr() << "\t"
              // since we seek for min Et and dynamic metric is looking for
              // max of its dynamic version, below for loging purposes the order
              // of division is swaped as it is basically inversion of the relative
              // dynamic metric
              << first.getEnergyTimeProd() / curr.getEnergyTimeProd() << "\t"
              << curr.checkPlusMetric(first, k);
    if (noNewLine) {
        sstream << std::flush;
    } else {
        sstream << "\n";
    }
    return sstream.str();
}

static inline
std::string logCurrentPowerLogtLine(
    double timeInMs,
    PowAndPerfResult& curr,
    const std::optional<PowAndPerfResult> reference = std::nullopt,
    double k = 2.0,
    const std::vector<double>* perSubdevicePowers = nullptr,
    bool noNewLine = false)
{
    std::stringstream sstream;
    sstream << timeInMs
            << std::fixed << std::setprecision(2)
            << "\t\t" << curr.appliedPowerCapInWatts_
            << "\t\t" << curr.averageCorePowerInWatts_
            << "\t\t " << curr.filteredPowerOfLimitedDomainInWatts_
            << "\t\t" << curr.energyInJoules_
            << "\t\t" << curr.instructionsCount_
            << std::fixed << std::setprecision(3)
            << "\t\t" << curr.getInstrPerJoule() * 1000
            << "\t\t" << curr.getEnergyTimeProd();
    if (reference.has_value())
    {
        double currRelativeENG = curr.getEnergyPerInstr() / reference.value().getEnergyPerInstr();
        // since we seek for min Et and dynamic metric is looking for
        // max of its dynamic version, below for loging purposes the order
        // of division is swaped as it is basically inversion of the relative
        // dynamic metric
        double currRelativeEDP = reference.value().getEnergyTimeProd() / curr.getEnergyTimeProd();
        sstream << "\t" << curr.getInstrPerSecond()
                << "\t" << curr.getInstrPerSecond() / reference.value().getInstrPerSecond()
                << "\t" << (std::isinf(currRelativeENG) || std::isnan(currRelativeENG) ? 1.0 : currRelativeENG)
                << "\t" << (std::isinf(currRelativeEDP) || std::isnan(currRelativeEDP) ? 1.0 : currRelativeEDP)
                << "\t" << curr.checkPlusMetric(reference.value(), k);
    }
    if (perSubdevicePowers && !perSubdevicePowers->empty())
    {
        for (double p : *perSubdevicePowers)
        {
            sstream << "\t" << p;
        }
    }
    if (noNewLine) {
        sstream << std::flush;
    } else {
        sstream << "\n";
    }
    return sstream.str();
}


class Logger
{
  public:
    Logger(std::string prefix)
    {
        const auto dir = generateUniqueDir(prefix);
        powerFileName_ = dir + "power_log.csv";
        resultFileName_ = dir + "result.csv";
        powerFile_.open(powerFileName_, std::ios::out | std::ios::trunc);
        resultFile_.open(resultFileName_, std::ios::out | std::ios::trunc);
        power_bout_ = std::make_unique<BothStream>(powerFile_);
        result_bout_ = std::make_unique<BothStream>(resultFile_);
        *power_bout_ << "#t[ms]\t\tP_cap[W]\t\tP_av[W]\t\tP_SMA[W]\t\tE[J]\t\tinstr[-]\t\tinst/En[1/J]\t\tEDP[Js]\tinstr/s\trel_ins/s\tdyn_rel_E\tdyn_rel_EDP\tdyn_EDS\n";
    }
    void logPowerLogLine(DeviceStateAccumulator& deviceState, PowAndPerfResult current, const std::optional<PowAndPerfResult> reference = std::nullopt)
    {
        // If device has multiple subdevices, also include their powers in the main CSV tail
        std::vector<double> subPowers;
        auto dev = deviceState.getDevice();
        if (dev && dev->getNumSubdevices() > 1)
        {
            ensurePerSubdevice(dev->getNumSubdevices());
            const auto t = deviceState.getTimeSinceObjectCreation();
            for (size_t i = 0; i < dev->getNumSubdevices(); ++i)
            {
                double p = dev->getCurrentPowerInWattsForSubdevice(i);
                subPowers.push_back(p);
                // minimal CSV: time, P_cap, P_av
                sub_power_files_[i] << t << "\t\t" << current.appliedPowerCapInWatts_ << "\t\t" << p << "\n";
            }
        }
        *power_bout_  << logCurrentPowerLogtLine(deviceState.getTimeSinceObjectCreation(), current, reference, 2.0, (subPowers.empty() ? nullptr : &subPowers));
    }
    void logToResultFile(std::stringstream& ss)
    {
        *result_bout_ << ss.str();
    }
    std::string getPowerFileName() const
    {
        return powerFileName_;
    }
    void flush() // might be useless
    {
        power_bout_->flush();
        result_bout_->flush();
    }
    std::string getResultFileName() const
    {
        return resultFileName_;
    }
    // Per-subdevice files
    void ensurePerSubdevice(size_t count)
    {
        if (sub_power_files_.size() >= count) return;
        sub_power_files_.resize(count);
        sub_power_names_.resize(count);
        for (size_t i = 0; i < count; ++i)
        {
            if (sub_power_names_[i].empty())
            {
                sub_power_names_[i] = generateSibling(powerFileName_, std::string("power_log_gpu") + std::to_string(i) + ".csv");
                sub_power_files_[i].open(sub_power_names_[i], std::ios::out | std::ios::trunc);
                sub_power_files_[i] << "#t[ms]\t\tP_cap[W]\t\tP_av[W]\t\tP_SMA[W]\t\tE[J]\t\tinstr[-]\t\tinst/En[1/J]\t\tEDP[Js]\n";
            }
        }
    }
    std::string getPerSubdeviceFileName(size_t idx) const { return sub_power_names_.at(idx); }
    ~Logger()
    {
        powerFile_.close();
        resultFile_.close();
    }
  private:
    std::string powerFileName_;
    std::ofstream powerFile_;
    std::string resultFileName_;
    std::ofstream resultFile_;
    std::unique_ptr<BothStream> power_bout_;
    std::unique_ptr<BothStream> result_bout_;
    std::vector<std::ofstream> sub_power_files_;
    std::vector<std::string> sub_power_names_;

    std::string generateUniqueDir(std::string prefix = "")
    {
        std::string dir = prefix + "_experiment_" +
            std::to_string(
                std::chrono::system_clock::to_time_t(
                      std::chrono::high_resolution_clock::now()));
        dir += "/";
        const int dir_err = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err) {
            printf("Error creating experiment result directory!\n");
            exit(1);
        }
        return dir;
    }
    static std::string generateSibling(const std::string& base, const std::string& name)
    {
        auto pos = base.find_last_of('/') ;
        if (pos == std::string::npos) return name;
        return base.substr(0, pos+1) + name;
    }
};