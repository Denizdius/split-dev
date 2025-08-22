/*
   Licensed under the Apache License, Version 2.0 (the "License");
*/

#include "devices/multi_cuda_device.hpp"

#include <sstream>
#include <algorithm>
#include <fstream>
#include <climits>
#include <cstdlib>
#include <sys/stat.h>

MultiCudaDevice::MultiCudaDevice(const std::vector<int>& deviceIds)
  : deviceIDs_(deviceIds)
{
  // CUDA driver init to be consistent with single GPU path
  CUresult cuRes = cuInit(0);
  if (cuRes != CUDA_SUCCESS) {
    std::cerr << "cuInit failed with code " << cuRes << "\n";
  }

  nvResult_ = nvmlInit();
  if (NVML_SUCCESS != nvResult_)
  {
    std::cerr << "Failed to initialize NVML: " << nvmlErrorString(nvResult_) << "\n";
    return;
  }
  nvResult_ = nvmlDeviceGetCount(&deviceCount_);
  if (NVML_SUCCESS != nvResult_)
  {
    std::cerr << "Failed to query device count: " << nvmlErrorString(nvResult_) << "\n";
    return;
  }
  initDeviceHandles();
  validateHomogeneousModel();
  defaultPowerLimitInWatts_.resize(deviceIDs_.size());
  for (size_t i = 0; i < deviceIDs_.size(); ++i)
  {
    unsigned currMw = 0;
    if (NVML_SUCCESS == nvmlDeviceGetEnforcedPowerLimit(deviceHandles_[deviceIDs_[i]], &currMw))
      defaultPowerLimitInWatts_[i] = static_cast<double>(currMw) / 1000.0;
    else
      defaultPowerLimitInWatts_[i] = 0.0;
  }
}

// Local helper to read an integer value from a file; returns -1 if not available
static inline long long readValueFromFile(const std::string& fileName)
{
  std::ifstream file(fileName.c_str());
  std::string line;
  long long value = -1;
  if (file.is_open())
  {
    while (std::getline(file, line))
    {
      value = std::atoll(line.c_str());
    }
    file.close();
  }
  return value;
}

void MultiCudaDevice::initDeviceHandles()
{
  deviceHandles_.resize(deviceCount_);
  for (unsigned i = 0; i < deviceCount_; ++i)
  {
    nvmlDevice_t h;
    nvmlReturn_t r = nvmlDeviceGetHandleByIndex(i, &h);
    if (r != NVML_SUCCESS)
    {
      std::cerr << "Failed to get handle for device " << i << ": " << nvmlErrorString(r) << "\n";
      continue;
    }
    deviceHandles_[i] = h;
  }
}

void MultiCudaDevice::validateHomogeneousModel() const
{
  // Ensure all selected devices share the same name
  if (deviceIDs_.empty()) return;
  char name0[96] = {0};
  if (nvmlDeviceGetName(deviceHandles_[deviceIDs_[0]], name0, sizeof(name0)) != NVML_SUCCESS) return;
  for (size_t i = 1; i < deviceIDs_.size(); ++i)
  {
    char name[96] = {0};
    if (nvmlDeviceGetName(deviceHandles_[deviceIDs_[i]], name, sizeof(name)) != NVML_SUCCESS) continue;
    if (std::string(name) != std::string(name0))
    {
      std::cerr << "Warning: Selected GPUs have different models; proceeding but behavior may vary.\n";
      break;
    }
  }
}

std::string MultiCudaDevice::getName() const
{
  if (deviceIDs_.empty()) return std::string("No GPU");
  char name[96] = {0};
  if (nvmlDeviceGetName(deviceHandles_[deviceIDs_[0]], name, sizeof(name)) != NVML_SUCCESS)
    return std::string("Unknown GPU");
  std::stringstream ss;
  ss << name << " x" << deviceIDs_.size();
  return ss.str();
}

std::pair<unsigned, unsigned> MultiCudaDevice::getMinMaxLimitInWatts() const
{
  // Intersect min/max across all selected GPUs
  unsigned globalMinW = 0;
  unsigned globalMaxW = UINT_MAX;
  for (int id : deviceIDs_)
  {
    unsigned minMw = 0, maxMw = 0;
    if (nvmlDeviceGetPowerManagementLimitConstraints(deviceHandles_[id], &minMw, &maxMw) == NVML_SUCCESS)
    {
      globalMinW = std::max(globalMinW, minMw / 1000);
      globalMaxW = std::min(globalMaxW, maxMw / 1000);
    }
  }
  if (globalMaxW < globalMinW) globalMaxW = globalMinW;
  return {globalMinW, globalMaxW};
}

double MultiCudaDevice::getPowerLimitInWatts() const
{
  // Return current limit of the first GPU (assume all equal when set via this class)
  if (deviceIDs_.empty()) return -1.0;
  unsigned currMw = 0;
  if (nvmlDeviceGetEnforcedPowerLimit(deviceHandles_[deviceIDs_[0]], &currMw) != NVML_SUCCESS) return -1.0;
  return static_cast<double>(currMw) / 1000.0;
}

void MultiCudaDevice::setPowerLimitInMicroWatts(unsigned long limitInMicroW)
{
  unsigned long limitInMilliWatts = limitInMicroW / 1000UL;
  for (int id : deviceIDs_)
  {
    nvmlReturn_t r = nvmlDeviceSetPowerManagementLimit(deviceHandles_[id], limitInMilliWatts);
    if (r != NVML_SUCCESS)
    {
      std::cerr << "Failed to set power limit " << limitInMilliWatts << " mW for GPU " << id
                << ": " << nvmlErrorString(r) << "\n";
    }
  }
}

void MultiCudaDevice::reset()
{
  // Mirror single-GPU behavior: initialize kernels counter file so perf reads don't block
  std::ofstream kernelCounterFile;
  kernelCounterFile.open("kernels_count", std::ios::out | std::ios::trunc);
  if (kernelCounterFile.is_open())
  {
    kernelCounterFile << "0";
    kernelCounterFile.close();
  }
}

double MultiCudaDevice::getCurrentPowerInWatts(std::optional<Domain>) const
{
  // Sum instantaneous power across all selected GPUs for logging
  double sumW = 0.0;
  for (int id : deviceIDs_)
  {
    unsigned powerMw = 0;
    if (nvmlDeviceGetPowerUsage(deviceHandles_[id], &powerMw) == NVML_SUCCESS)
      sumW += static_cast<double>(powerMw) / 1000.0;
  }
  return sumW;
}

double MultiCudaDevice::getCurrentPowerInWattsForSubdevice(size_t index) const
{
  if (index >= deviceIDs_.size()) return 0.0;
  unsigned powerMw = 0;
  if (nvmlDeviceGetPowerUsage(deviceHandles_[deviceIDs_[index]], &powerMw) != NVML_SUCCESS)
    return 0.0;
  return static_cast<double>(powerMw) / 1000.0;
}

unsigned long long int MultiCudaDevice::getPerfCounter() const
{
  // Reuse the same perf counter source (injection library aggregates at process level)
  // Non-blocking: if the file is not present yet, return 0
  long long kernelsCountTmp = readValueFromFile("./kernels_count");
  if (kernelsCountTmp < 0) {
    return 0ULL;
  }
  return static_cast<unsigned long long>(kernelsCountTmp);
}

void MultiCudaDevice::restoreDefaultLimits()
{
  for (size_t i = 0; i < deviceIDs_.size(); ++i)
  {
    unsigned long mw = static_cast<unsigned long>(defaultPowerLimitInWatts_[i] * 1000.0);
    nvmlReturn_t r = nvmlDeviceSetPowerManagementLimit(deviceHandles_[deviceIDs_[i]], mw);
    if (r != NVML_SUCCESS)
    {
      std::cerr << "Failed to restore default power limit for GPU " << deviceIDs_[i]
                << ": " << nvmlErrorString(r) << "\n";
    }
  }
}


