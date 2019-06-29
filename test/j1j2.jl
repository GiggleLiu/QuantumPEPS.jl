# define the target model
using Yao
using QuantumPEPS
model = J1J2(6, 4; J2=0.5, periodic=false)
config = QPEPSConfig(4, 2, 4, 1)
runtime = QPEPSRunTime(config)
samples = gensample(config, runtime, Z; nbatch=100)

energy(samples, model)
