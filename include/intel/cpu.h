
#ifndef Intel_CPU_H
#define Intel_CPU_H

namespace intel {

// test optional cpu features
// neon = armv7 neon or aarch64 asimd
int cpu_support_arm_neon();
// vfpv4 = armv7 fp16 + fma
int cpu_support_arm_vfpv4();
// asimdhp = aarch64 asimd half precision
int cpu_support_arm_asimdhp();

// cpu info
int get_cpu_count();

// bind all threads on little clusters if powersave enabled
// affacts HMP arch cpu like ARM big.LITTLE
// only implemented on android at the moment
// switching powersave is expensive and not thread-safe
// 0 = all cores enabled(default)
// 1 = only little clusters enabled
// 2 = only big clusters enabled
// return 0 if success for setter function
int get_cpu_powersave();
int set_cpu_powersave(int powersave);

// misc function wrapper for openmp routines
int get_omp_num_threads();
void set_omp_num_threads(int num_threads);

int get_omp_dynamic();
void set_omp_dynamic(int dynamic);

} // namespace intel

#endif // Intel_CPU_H
