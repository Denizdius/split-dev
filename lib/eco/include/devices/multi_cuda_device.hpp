/*
   Copyright 2025

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

#include <vector>
#include <string>
#include <optional>
#include <iostream>

#include "devices/abstract_device.hpp"

#include <cuda.h>
#include <nvml.h>

class MultiCudaDevice : public Device
{
  public:
    /// When \p asyncIndependentPerGpuCaps is true (DEPO --async with multiple GPUs), each GPU can use a different power cap.
    explicit MultiCudaDevice(const std::vector<int>& deviceIds, bool asyncIndependentPerGpuCaps = false);
    ~MultiCudaDevice() override = default;

    // Device interface
    std::string getName() const override;
    std::pair<unsigned, unsigned> getMinMaxLimitInWatts() const override;
    double getPowerLimitInWatts() const override;
    void setPowerLimitInMicroWatts(unsigned long limitInMicroW) override;
    void setPowerLimitsPerGpuMicroWatts(const std::vector<unsigned long>& microWattsPerSubdevice) override;
    bool usesIndependentSubdevicePowerCaps() const override { return asyncIndependentPerGpuCaps_; }
    std::vector<unsigned long> getCurrentPerGpuCapsMicroWatts() const override { return currentCapsMicroW_; }
    void reset() override;
    double getCurrentPowerInWatts(std::optional<Domain> = std::nullopt) const override;
    unsigned long long int getPerfCounter() const override;
    void triggerPowerApiSample() override {}
    void restoreDefaultLimits() override;
    std::string getDeviceTypeString() const override { return "gpu"; }
    size_t getNumSubdevices() const override { return deviceIDs_.size(); }
    double getCurrentPowerInWattsForSubdevice(size_t index) const override;
    std::string getSubdeviceLabel(size_t index) const override { return std::string("gpu") + std::to_string(deviceIDs_.at(index)); }

    /// Run stock single-GPU search on one GPU: \p baselineCaps holds fixed limits for all GPUs; only index \p focusIndex is swept.
    void beginPerGpuSearchSession(size_t focusIndex, const std::vector<unsigned long>& baselineCapsMicroW);
    void endPerGpuSearchSession();

  private:
    void applyPerGpuVectorMicroWatts_(const std::vector<unsigned long>& capsMicroW);
    void initDeviceHandles();
    void validateHomogeneousModel() const;

    nvmlReturn_t nvResult_ {NVML_SUCCESS};
    unsigned int deviceCount_ {0};
    std::vector<int> deviceIDs_;
    std::vector<nvmlDevice_t> deviceHandles_;
    std::vector<double> defaultPowerLimitInWatts_;
    bool asyncIndependentPerGpuCaps_ {false};
    std::vector<unsigned long> currentCapsMicroW_;
    bool inPerGpuSearchSession_ {false};
    size_t searchFocusIndex_ {0};
    std::vector<unsigned long> searchBaselineCapsMicroW_;
};


