// host_check.cpp — compile & test the device-independent bit logic from mito4_kernel.cu
// This does NOT need CUDA; it validates the pack/unpack/switch math matches Python.
#include <cstdio>
#include <cstdint>
#include <cassert>

static inline uint32_t f_energy(uint32_t w){ return  w        & 0xFFu; }
static inline uint32_t f_thresh(uint32_t w){ return (w >> 8)  & 0xFFu; }
static inline uint32_t f_age   (uint32_t w){ return (w >> 16) & 0xFFu; }
static inline uint32_t f_alive (uint32_t w){ return (w >> 24) & 0x1u;  }
static inline uint32_t f_gen   (uint32_t w){ return (w >> 25) & 0x7Fu; }
static inline uint32_t mk(uint32_t e,uint32_t t,uint32_t a,uint32_t l,uint32_t g){
    if(e>255)e=255; if(a>255)a=255; if(g>127)g=127;
    return (e & 0xFFu) | ((t & 0xFFu)<<8) | ((a & 0xFFu)<<16) | ((l & 0x1u)<<24) | ((g & 0x7Fu)<<25);
}

int main(){
    // round-trip parity with the Python seed 0x01005a0b (E=11,T=90,gen=0)
    uint32_t w = mk(11,90,0,1,0);
    printf("word = 0x%08x  E=%u T=%u A=%u L=%u G=%u\n",
           w, f_energy(w), f_thresh(w), f_age(w), f_alive(w), f_gen(w));
    assert(w == 0x01005a0bu);
    assert(f_energy(w)==11 && f_thresh(w)==90 && f_alive(w)==1 && f_gen(w)==0);

    // switch parity: divide iff E-T>=0
    assert(((int)f_energy(mk(200,90,0,1,0)) - (int)f_thresh(mk(200,90,0,1,0))) >= 0);  // divide
    assert(((int)f_energy(mk(50,90,0,1,0))  - (int)f_thresh(mk(50,90,0,1,0)))  <  0);  // no divide

    // field clamping
    assert(f_energy(mk(999,0,0,1,0))==255);
    assert(f_gen(mk(0,0,0,1,999))==127);

    // mitosis halving + gen increment
    uint32_t parent = mk(200,90,5,1,3);
    uint32_t child  = mk(f_energy(parent)/2, f_thresh(parent), 0, 1, f_gen(parent)+1);
    assert(f_energy(child)==100 && f_thresh(child)==90 && f_gen(child)==4 && f_age(child)==0);

    printf("ALL HOST-SIDE BIT LOGIC CHECKS PASSED (parity with Python reference)\n");
    return 0;
}
