# Anotation

This is an open source work-in-progress repository collecting a set of software tools for power monitoring and controlling in HPC systems.
Some features might change in the future, some features might be removed when part of the technology used here become deprecated.

# SPLiT

Welcome to Software Power Limiting Tools repository - an open source collection of tools designed
for supporting energy-aware high-performance computing.
So far the repository supports Intel based CPUs and NVIDIA GPUs.

## The tools available at the moment:
1. Static Energy Profiler (StEP) `./build/StEP`
2. Dynamic Energy-Performance Optimizer `./build/DEPO`

### Static Energy Profiler (StEP)
This tool is designed for static exploration of the energy characteristic of the given Device.
The device types supported at the moment are:
- Intel CPU
- Intel XPU (Ponte Vecchio)
- NVIDIA GPU

The tool is designed for automatic examination of power limits impact on energy consumption and performance
of application executed on examined device.
Basically, the tool automates the subsequent execution of the application with different power caps (limits)
and as a result produces the report of the best power cap with respect to 3 predefined target optimization metrics.
Since Intel CPUs don't have clear minimal power limit value the tool it is exploring the power limits range between
the idle consumption power limit level (measured prior to application execution) and default power cap value
(usually the same as maximal available power limit value). The power limits are examined with Linear Search method,
starting from the highest value, with decrement defined percentage-wise by the user in the `params.conf` file.

The details of the tool for CPU might be found in the paper:

>Krzywaniak, A., Czarnul, P., & Proficz, J. (2019).
>
>*Extended investigation of performance-energy trade-offs under power capping in HPC environments.*
>
>https://doi.org/10.1109/hpcs48598.2019.9188149

For GPU the power limits are explored between max and min value (which are defined by NVIDIA and available to obtain
via NVML API). The power decrement is fixed for 5W.

The details of the tool for GPU might be found in the paper:
>Krzywaniak, A., Czarnul, P., & Proficz, J. (2022).
>
>*GPU Power Capping for Energy-Performance Trade-Offs in Training of Deep Convolutional Neural Networks for Image Recognition.*
>
>https://doi.org/10.1007/978-3-031-08751-6_48


### Dynamic Energy-Performance Optimizer (DEPO)
This tool is designed for dynamic exploration and selection of the power cap according to selected target optimization metric. The tool shall be launched with the application for which the energy and performance
shall be optimized by adjusting the power cap value with respect to the selected target metric.
The tool is able to perform the search with Linear Search Algorithm or with Golden Section Search Algorithm.
The details of the tool for CPU may be found in the paper:

>Krzywaniak, A., Czarnul, P., & Proficz, J. (2022).
>
>*DEPO: A dynamic energy‐performance optimizer tool for automatic power capping for energy efficient high‐performance computing. SOFTWARE-PRACTICE & EXPERIENCE, 52, 2598-2634.*
>
>https://doi.org/10.1002/spe.3139


The details of the tool for GPU may be found in the paper:

>Adam Krzywaniak, Paweł Czarnul, Jerzy Proficz (2023)
>
> *Dynamic GPU power capping with online performance tracing for energy efficient GPU computing using DEPO tool, Future Generation Computer Systems, Volume 145, 2023, Pages 396-414, ISSN 0167-739X.*
>
>https://doi.org/10.1016/j.future.2023.03.041.


# Build and usage
It is recommended to create a build directory.
```
mkdir build
cd build
cmake ..
make
```

### Building for Intel XPU
#### Make build directory
```
mkdir build
cd build
```
#### Configure
```
cmake -DWITH_XPU=ON ..
```
###### If you have Level Zero installed in custom location, you can specify it with LIBZE_LOADER_PATH:

```
cmake ../ -DWITH_XPU=ON -DLIBZE_LOADER_PATH=<custom directory with libze_loader.so>
```

#### Testing
To test your configuration you can run unit tests (with super user priviliges):
```bash
sudo ctest -V
```
or with logging all messages to console (where log_level is: info/warn/debug/trace/err/critical)
```bash
sudo SPDLOG_LEVEL=trace ctest -V
```

# Known dependencies
```
sudo apt update && sudo apt install build-essential cmake gnuplot
sudo apt install libboost-all-dev graphviz
sudo apt install libyaml-cpp-dev
```

# Exemplary usage
Note: the power limiting feature requires root privileges in Ubuntu OS, hence below commands are executed as `sudo` user.

## StEP
`sudo ./build/apps/StEP/StEP ./minibenchmarks/openmp/fft 16384 25`


Above command shall run exemplary FFT application with StEP for CPU and produce
as a result `cpu_experiment_*` folder with `.csv` logs and visualised `.png`
power log similar to below one:

![exemplary power log step](docs/power_log_step.png)

and StEP tool results visualised as below:

![exemplary step result](docs/result_step.png)
![exemplary step result et](docs/result_step_et.png)

### Using power capping  instead of current capping for Intel XPU
By default Current capping is used for XPU. In case you want to test power capping, you need to set the `USE_AMPERES` environment variable to `0` before running the application. For example:
`sudo USE_AMPERES=0 ./build/apps/StEP/StEP <cmdline of your Intel XPU workload>`

## DEPO
`sudo ./build/apps/DEPO/DEPO --ls --en ./minibenchmarks/openmp/fft 1024 300`

Above command shall run exemplary FFT application with DEPO and produce
as a result `cpu_experiment_*` folder with `.csv` logs and visualised `.png`
power log. See details in the next section.

### DEPO configuration

One may modify the execution parameters in `config.yaml` file or with command line parameters which may be listed
with `DEPO --help`.

The parameters in the `config.yaml` file are documented in comments.

### Available search modes in DEPO
In DEPO there are several optimization modes available:
1. **Just power sampling**, which launches the application and monitors and reports power and energy consumption when finished, available when `--no-tuning` parameter is passed.
![exemplary depo result sampling](docs/result_depo_sampling.png)
2. **Single immediate tuning**, which launches the Tuing Phase as soon as the optimized device activity is detected, available with `config.yaml` parameters set to: `repeatTuningPeriodInSec: 0` and `doWaitPhase: 0`.
![exemplary depo result single immediate](docs/result_depo_single_immediate.png)
3. **Single tuning with wait**, which launches Tuing Phase after SMA based Power filter detects stable average power consumption, available with `config.yaml` parameters set to: `repeatTuningPeriodInSec: 0` and `doWaitPhase: 1`.
![exemplary depo result single wait](docs/result_depo_single_wait.png)
4. **Periodic immediate tuning**, which launches the Tuning Phase as soon as the optimized device activity is detected and repeats the tuning phase after a period defined in seconds with `repeatTuningPeriodInSec: 30` (for 30s execution with selected power cap before next Tuning Phase). Assuming `doWaitPhase: 0`.
![exemplary depo result periodic immediate](docs/result_depo_periodic_immediate.png)
5. **Periodic tuning with wait**, which adds Wait Phase before first Tuning Phase, available with `config.yaml` parameters set to: `repeatTuningPeriodInSec: 30` (for 30s period before next Tuing Phase) and `doWaitPhase: 1`.
![exemplary depo result periodic wait](docs/result_depo_periodic_wait.png)

#### Linear Search algorithm
For any mode one may run DEPO with Linear Search algorithm as well:
![exemplary depo result periodic immediate ls](docs/result_depo_ls_periodic.png)

### Experimental asynchronous Tuning in DEPO

There is also a way of triggering a tuning phase on demand with external signal.
For now it works with application executed in any of available DEPO modes besides "just sampling". It will be fixed soon.
The feature allows triggering the next Tuning Phase with external trigger using specific file modification.
All one has to do to trigger asynchronously the next Tuing Phase is to execute `touch /tmp/trigger_file` during execution of DEPO with selected application.
![exemplary depo result with external trigger](docs/result_depo_external_trigger.png)

# Adding support for other devices
If one wish to add support for other CPU vendors, other GPU vendors or other compute devices they have to make sure that the target device provides:
1. an API for monitoring the power or energy consumption
2. an API for monitoring the performance of the compute device (e.g., executed instructions per second counters)
3. an API for controlling the power limits

The new HW support may be added by preparing a `NewDevice` class inherited from the `Device` class, which would implement the interface required for using the `Device` by `Eco` class.
Next step would be adding the `NewDevice` option in the DEPO application source file, i.e., `src/apps/DynamicECO.cpp` or writing own DEPO program with just the `NewDevice` class.

### Current classes and dependencies diagram
![DEPO class diagram](docs/depo_class_diagram.png)

# Related works
If you find this code usefull please cite any of our papers which contributed to this codebase:

1. Static Energy Profiler (StEP) for CPU:

        @INPROCEEDINGS{9188149,
        author={Krzywaniak, Adam and Czarnul, Pawel and Proficz, Jerzy},
        booktitle={2019 International Conference on High Performance Computing & Simulation (HPCS)},
        title={Extended investigation of performance-energy trade-offs under power capping in HPC environments},
        year={2019},
        pages={440-447},
        doi={10.1109/HPCS48598.2019.9188149}
        }

2. Static Energy Profiler (StEP) for GPU:

        @inproceedings{10.1007/978-3-031-08751-6_48,
        author = {Krzywaniak, Adam and Czarnul, Pawel and Proficz, Jerzy},
        booktitle = {Computational Science -- ICCS 2022},
        isbn = {978-3-031-08751-6},
        pages = {667--681},
        publisher = {Springer International Publishing},
        title = {GPU Power Capping for Energy-Performance Trade-Offs in Training of Deep Convolutional Neural Networks for Image Recognition},
        year = {2022}
        }

3. Dynamic Energy-Performance Optimizer (DEPO) for CPU:

        @article{https://doi.org/10.1002/spe.3139,
        author = {Krzywaniak, Adam and Czarnul, Paweł and Proficz, Jerzy},
        title = {{DEPO}: A dynamic energy-performance optimizer tool for automatic power capping for energy efficient high-performance computing},
        journal = {Software: Practice and Experience},
        volume = {52},
        number = {12},
        pages = {2598-2634},
        keywords = {automatic power capping, green computing, HPC, performance-energy trade-off, software tools},
        doi = {https://doi.org/10.1002/spe.3139},
        year = {2022}
        }

4. Dynamic Energy-Performance Optimizer (DEPO) for GPU:

        @article{KRZYWANIAK2023396,
        author = {Adam Krzywaniak and Pawe{\l} Czarnul and Jerzy Proficz},
        doi = {https://doi.org/10.1016/j.future.2023.03.041},
        issn = {0167-739X},
        journal = {Future Generation Computer Systems},
        keywords = {Energy-aware computing, High-performance computing, Green computing, Machine learning, GPU energy optimization},
        pages = {396-414},
        title = {Dynamic GPU power capping with online performance tracing for energy efficient GPU computing using DEPO tool},
        volume = {145},
        year = {2023},
        }


# Some notes to be expanded in future

https://developer.nvidia.com/cuda-downloads


DEPO GPU [requires CUDA compute capabilities 7.0 or higher](https://docs.nvidia.com/cupti/main/main.html#cupti-profiling-api), what means that architectures older than Volta (e.g., Pascal) are not supported.
