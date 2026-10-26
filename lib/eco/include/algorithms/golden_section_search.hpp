/*
   Copyright 2024, Adam Krzywaniak.

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

#include "algorithms/abstract_search_algorithm.hpp"


class GoldenSectionSearchAlgorithm : public SearchAlgorithm
{
  public:
    unsigned operator() (
      std::shared_ptr<Device> device,
      DeviceStateAccumulator& deviceState,
      Trigger& trigger,
      TargetMetric metric,
      const PowAndPerfResult& reference,
      int& procStatus,
      int childProcID,
      int powerSamplingPeriodInMilliSeconds,
      int tuningTimeWindowInMilliSeconds,
      Logger& logger) const
    {
        const auto [minLimitInWatts, maxLimitInWatts] = device->getMinMaxLimitInWatts();
        int EPSILON = (maxLimitInWatts - minLimitInWatts) * 1e6 / 25;

        int a = minLimitInWatts * 1e6; // micro watts
        int b = maxLimitInWatts * 1e6; // micro watts

        int leftCandidateInMicroiWatts = b - int(PHI * (b - a));
        int rightCandidateInMicroWatts = a + int(PHI * (b - a));

        bool measureL = true;
        bool measureR = true;

        PowAndPerfResult tmp = reference;
        const int repeats = device->searchSampleRepeats() < 1 ? 1 : device->searchSampleRepeats();
        auto measureCandidate = [&](int limitMicro) {
            device->setPowerLimitInMicroWatts(static_cast<unsigned long>(limitMicro));
            PowAndPerfResult acc;
            for (int r = 0; r < repeats; ++r)
            {
                auto sample = sampleAndAccumulatePowAndPerfForGivenPeriod(
                    tuningTimeWindowInMilliSeconds * 1000,
                    powerSamplingPeriodInMilliSeconds,
                    deviceState,
                    trigger,
                    procStatus,
                    childProcID,
                    logger);
                logger.logPowerLogLine(deviceState, sample, reference);
                if (r == 0) acc = sample;
                else acc += sample;
                waitpid(childProcID, &procStatus, WNOHANG);
                if (!procStatus) break;
            }
            return acc;
        };

        while ((b - a) > EPSILON)
        {
          logCurrentRangeGSS(a, leftCandidateInMicroiWatts/1000, rightCandidateInMicroWatts/1000, b);
          auto fL = tmp;
          if (measureL)
          {
            fL = measureCandidate(leftCandidateInMicroiWatts);
          }
          auto fR = tmp;
          if (measureR)
          {
            fR = measureCandidate(rightCandidateInMicroWatts);
          }

          if (!fL.isRightBetter(fR, metric)) {
            // choose subrange [a, rightCandidateInMilliWatts]
            b = rightCandidateInMicroWatts;
            rightCandidateInMicroWatts = leftCandidateInMicroiWatts;
            tmp = fL;
            measureR = false;
            measureL = true;
            leftCandidateInMicroiWatts = b - int(PHI * (b - a));
          } else {
            // choose subrange [leftCandidateInMilliWatts, b]
            a = leftCandidateInMicroiWatts;
            leftCandidateInMicroiWatts = rightCandidateInMicroWatts;
            tmp = fR;
            measureR = true;
            measureL = false;
            rightCandidateInMicroWatts = a + int(PHI * (b - a));
          }
          waitpid(childProcID, &procStatus, WNOHANG);
          if (!procStatus) break;
        }
        return (a + b) / 2;
    }
    static constexpr float PHI {(sqrt(5) - 1) / 2}; // this is equal 0.618 and it is reverse of 1.618
  private:
    void logCurrentRangeGSS(int a, int leftCandidateInMilliWatts, int rightCandidateInMilliWatts, int b) const
    {
        std::cout << "#--------------------------------\n"
                  << "# Current GSS range: |"
                  << a << " "
                  << leftCandidateInMilliWatts << " "
                  << rightCandidateInMilliWatts << " "
                  << b << "|\n"
                  << "#--------------------------------\n";
    }
};
