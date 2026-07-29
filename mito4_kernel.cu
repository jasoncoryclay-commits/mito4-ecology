// mito4_kernel.cu — MITO-4 ECOLOGY on the GPU (H100-ready)
// Grid-resident spatial variant of the Minimal Integer Threshold Organism.
//
// Build:   nvcc -O3 -arch=sm_90 mito4_kernel.cu -o mito4          (sm_90 = Hopper/H100)
// Run:     ./mito4 <H> <W> <ticks> <seed> <log_every> [dump_every]
//          dump_every > 0  -> write snapshot PGM images every N ticks (for H4)
//
// Design (matches mito4_ecology.py bit-for-bit in intent):
//   * State is a FIXED HxW array of uint32 organisms -> no dynamic allocation,
//     no stream-compaction. (Grid-resident placement replaces list growth.)
//   * Resource field is a FIXED HxW array of float.
//   * One CUDA thread per grid cell. All updates are branch-light.
//   * Mitosis places a daughter into an empty von-Neumann neighbor, resolving
//     the write-race with atomicCAS on the alive bit in 4 deterministic
//     directional passes (N,S,W,E) — same priority order as the CPU reference.
//
// 4-byte word layout (little-endian, GPU-native):
//   bits 0-7 energy | 8-15 threshold | 16-23 age | 24 alive | 25-31 generation

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>   // std::swap
#include <cuda_runtime.h>

// ---- bit-field helpers -----------------------------------------------------
__host__ __device__ __forceinline__ uint32_t f_energy(uint32_t w){ return  w        & 0xFFu; }
__host__ __device__ __forceinline__ uint32_t f_thresh(uint32_t w){ return (w >> 8)  & 0xFFu; }
__host__ __device__ __forceinline__ uint32_t f_age   (uint32_t w){ return (w >> 16) & 0xFFu; }
__host__ __device__ __forceinline__ uint32_t f_alive (uint32_t w){ return (w >> 24) & 0x1u;  }
__host__ __device__ __forceinline__ uint32_t f_gen   (uint32_t w){ return (w >> 25) & 0x7Fu; }

__host__ __device__ __forceinline__ uint32_t mk(uint32_t e,uint32_t t,uint32_t a,uint32_t l,uint32_t g){
    if(e>255)e=255; if(a>255)a=255; if(g>127)g=127;
    return (e & 0xFFu) | ((t & 0xFFu)<<8) | ((a & 0xFFu)<<16) | ((l & 0x1u)<<24) | ((g & 0x7Fu)<<25);
}

// ---- simulation parameters (mirror the CPU reference) ----------------------
#define REGEN        8.0f
#define RES_CAP      255.0f
#define DIFFUSION    0.12f
#define UPKEEP       30
#define HARVEST_FRAC 0.6f
#define MUT_P        0.03f   // daughter threshold mutation probability

__host__ __device__ __forceinline__ int idx(int y,int x,int H,int W){
    // torus wrap
    y = (y + H) % H; x = (x + W) % W; return y*W + x;
}

// cheap per-cell/per-tick hash RNG (deterministic, seed-driven)
// __host__ __device__ so main() can use it to seed the initial grid on the host.
__host__ __device__ __forceinline__ uint32_t hash_u32(uint32_t a){
    a ^= a>>16; a *= 0x7feb352du; a ^= a>>15; a *= 0x846ca68bu; a ^= a>>16; return a;
}
__device__ __forceinline__ float rand01(uint32_t seed,uint32_t cell,uint32_t tick,uint32_t salt){
    return (hash_u32(seed*2654435761u ^ cell*40503u ^ tick*2246822519u ^ salt*668265263u) & 0xFFFFFF) / (float)0x1000000;
}

// ---- KERNEL 1: resource diffusion + regen ---------------------------------
__global__ void k_diffuse(const float* res_in, float* res_out, int H, int W){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;
    float c = res_in[idx(y,x,H,W)];
    float nb = res_in[idx(y-1,x,H,W)] + res_in[idx(y+1,x,H,W)]
             + res_in[idx(y,x-1,H,W)] + res_in[idx(y,x+1,H,W)];
    float v = c*(1.0f-DIFFUSION) + DIFFUSION*0.25f*nb;
    v += REGEN; if(v>RES_CAP) v=RES_CAP;
    res_out[idx(y,x,H,W)] = v;
}

// ---- KERNEL 2: harvest + switch + upkeep/death (in place on organisms) -----
// Writes: updated grid (survivors), and a per-cell 'wants_divide' flag + halved
// energy staged for daughters. Daughters are placed in KERNEL 3.
__global__ void k_metabolize(uint32_t* grid, float* res, uint8_t* wants, int H,int W){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;
    int i = y*W + x;
    uint32_t w = grid[i];
    wants[i] = 0;
    if(!f_alive(w)) return;

    int e = (int)f_energy(w);
    int t = (int)f_thresh(w);
    int a = (int)f_age(w);
    int g = (int)f_gen(w);

    // local harvest
    float r = res[i];
    int harvest = (int)floorf(r * HARVEST_FRAC);
    if(harvest > 255-e) harvest = 255-e;
    if(harvest < 0) harvest = 0;
    e += harvest;
    r -= harvest; if(r<0) r=0; res[i] = r;
    a = a<255 ? a+1 : 255;

    if(e - t >= 0){
        // wants to divide: parent halves energy, gen+1, stages daughter in KERNEL 3
        wants[i] = 1;
        grid[i] = mk(e/2, t, a, 1, g+1);   // provisional; blocked-division handled in K3 note
    } else {
        e -= UPKEEP;
        if(e <= 0){ grid[i] = 0u; }                    // starved -> empty
        else      { grid[i] = mk(e, t, a, 1, g); }     // persist
    }
}

// ---- KERNEL 3: mitosis placement into empty neighbor (atomic, 4 passes) ----
// pass direction (dy,dx). Each wanting parent tries to claim the neighbor cell.
// We claim by atomicCAS(grid[nbr], 0 -> daughter_word). First writer wins ->
// deterministic-ish; we run fixed N,S,W,E order to match the CPU reference.
__global__ void k_divide(uint32_t* grid, uint8_t* wants, int H,int W,
                         int dy,int dx, uint32_t seed, uint32_t tick){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;
    int i = y*W + x;
    if(!wants[i]) return;                 // not (or no longer) a dividing parent
    uint32_t p = grid[i];
    if(!f_alive(p)) return;

    int ni = idx(y+dy, x+dx, H, W);
    // daughter inherits parent's (already-halved) energy + threshold, age 0, gen from parent
    uint32_t de = f_energy(p);
    uint32_t dt = f_thresh(p);
    uint32_t dg = f_gen(p);
    // optional mutation
    if(rand01(seed,(uint32_t)ni,tick,7u) < MUT_P){
        float u = rand01(seed,(uint32_t)ni,tick,11u);
        int delta = (u<0.5f)? -1 : 1;
        int nt = (int)dt + delta; if(nt<1) nt=1; if(nt>255) nt=255; dt=(uint32_t)nt;
    }
    uint32_t daughter = mk(de, dt, 0, 1, dg);
    // claim empty neighbor atomically
    uint32_t old = atomicCAS(&grid[ni], 0u, daughter);
    if(old == 0u){
        wants[i] = 0;                     // satisfied -> stop trying in later passes
    }
    // if neighbor was occupied, parent keeps trying next direction; if all 4 fail,
    // parent simply persists as already written in K2 (division "blocked" — note:
    // its energy was halved provisionally; see PARITY NOTE in README).
}

// ---- reduction for stats (population + threshold histogram) ----------------
__global__ void k_count(const uint32_t* grid, unsigned long long* alive_out,
                        unsigned long long* thr_sum, unsigned int* thr_present,
                        int H,int W){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if(x>=W||y>=H) return;
    uint32_t w = grid[y*W+x];
    if(f_alive(w)){
        atomicAdd(alive_out, 1ull);
        atomicAdd(thr_sum, (unsigned long long)f_thresh(w));
        atomicOr(&thr_present[f_thresh(w)>>5], 1u << (f_thresh(w)&31)); // presence bitset (256 bits -> 8 u32)
    }
}

// ---------------------------------------------------------------------------
#define CK(call) do{ cudaError_t e=(call); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)

// ---- snapshot: write a downsampled threshold/occupancy map as a binary PGM ---
// Produces an SxS grayscale image where pixel = mean threshold of live cells in
// that block (0 if empty). Lets you SEE fronts / patches / waves for H4.
static void dump_pgm(const uint32_t* grid, int H, int W, int seed, int tick){
    const int S = 512;                        // output image is S x S (downsampled)
    int bh = (H + S - 1) / S, bw = (W + S - 1) / S;
    unsigned char* img = (unsigned char*)calloc((size_t)S*S, 1);
    for(int by=0; by<S; by++){
        for(int bx=0; bx<S; bx++){
            long sum=0, cnt=0;
            for(int y=by*bh; y<(by+1)*bh && y<H; y++)
                for(int x=bx*bw; x<(bx+1)*bw && x<W; x++){
                    uint32_t w = grid[(size_t)y*W+x];
                    if((w>>24)&1u){ sum += (w>>8)&0xFFu; cnt++; }
                }
            img[by*S+bx] = cnt ? (unsigned char)(sum/cnt) : 0;
        }
    }
    char fn[128];
    snprintf(fn,sizeof(fn),"snapshot_seed%d_t%05d.pgm",seed,tick);
    FILE* f = fopen(fn,"wb");
    if(f){ fprintf(f,"P5\n%d %d\n255\n",S,S); fwrite(img,1,(size_t)S*S,f); fclose(f);
           fprintf(stderr,"# wrote %s\n",fn); }
    free(img);
}

int main(int argc, char** argv){
    int H = argc>1?atoi(argv[1]):4096;
    int W = argc>2?atoi(argv[2]):4096;
    int TICKS = argc>3?atoi(argv[3]):2000;
    uint32_t seed = argc>4?(uint32_t)atoi(argv[4]):1u;
    int log_every = argc>5?atoi(argv[5]):100;
    int dump_every = argc>6?atoi(argv[6]):0;    // 0 = no image snapshots
    size_t N = (size_t)H*W;

    printf("MITO-4 ECOLOGY on GPU  |  %dx%d = %zu cells  |  %d ticks  |  seed %u\n",
           H,W,N,TICKS,seed);

    // host init
    uint32_t* h_grid = (uint32_t*)calloc(N,sizeof(uint32_t));
    float*    h_res  = (float*)malloc(N*sizeof(float));
    for(size_t i=0;i<N;i++) h_res[i] = RES_CAP*0.5f;
    // seed ~2% organisms with varied thresholds
    for(size_t i=0;i<N;i++){
        uint32_t r = hash_u32((uint32_t)i ^ (seed*2654435761u));
        if((r & 0x3F)==0){ // ~1/64 density
            uint32_t e = 20 + (r>>6)%40;
            uint32_t t = 70 + (r>>12)%60;
            h_grid[i] = mk(e,t,0,1,0);
        }
    }

    // device buffers
    uint32_t *d_grid; float *d_res, *d_res2; uint8_t *d_wants;
    CK(cudaMalloc(&d_grid, N*sizeof(uint32_t)));
    CK(cudaMalloc(&d_res,  N*sizeof(float)));
    CK(cudaMalloc(&d_res2, N*sizeof(float)));
    CK(cudaMalloc(&d_wants,N*sizeof(uint8_t)));
    CK(cudaMemcpy(d_grid,h_grid,N*sizeof(uint32_t),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_res, h_res, N*sizeof(float),   cudaMemcpyHostToDevice));

    unsigned long long *d_alive,*d_thrsum; unsigned int* d_present;
    CK(cudaMalloc(&d_alive,sizeof(unsigned long long)));
    CK(cudaMalloc(&d_thrsum,sizeof(unsigned long long)));
    CK(cudaMalloc(&d_present,8*sizeof(unsigned int)));

    dim3 blk(16,16);
    dim3 grd((W+15)/16,(H+15)/16);
    const int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};

    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    printf("tick,alive,mean_thr,lineages\n");
    for(int t=0;t<TICKS;t++){
        // 1) diffuse resource (ping-pong)
        k_diffuse<<<grd,blk>>>(d_res,d_res2,H,W);
        std::swap(d_res,d_res2);
        // 2) harvest + switch + death
        k_metabolize<<<grd,blk>>>(d_grid,d_res,d_wants,H,W);
        // 3) mitosis placement in 4 deterministic passes
        for(int d=0;d<4;d++)
            k_divide<<<grd,blk>>>(d_grid,d_wants,H,W,dirs[d][0],dirs[d][1],seed,(uint32_t)t);

        if(t%log_every==0 || t==TICKS-1){
            CK(cudaMemset(d_alive,0,sizeof(unsigned long long)));
            CK(cudaMemset(d_thrsum,0,sizeof(unsigned long long)));
            CK(cudaMemset(d_present,0,8*sizeof(unsigned int)));
            k_count<<<grd,blk>>>(d_grid,d_alive,d_thrsum,d_present,H,W);
            unsigned long long alive,thrsum; unsigned int present[8];
            CK(cudaMemcpy(&alive,d_alive,sizeof(alive),cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(&thrsum,d_thrsum,sizeof(thrsum),cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(present,d_present,8*sizeof(unsigned int),cudaMemcpyDeviceToHost));
            int lineages=0; for(int k=0;k<8;k++) lineages+=__builtin_popcount(present[k]);
            double mthr = alive? (double)thrsum/alive : 0.0;
            printf("%d,%llu,%.2f,%d\n",t,alive,mthr,lineages);
            fflush(stdout);
            if(alive==0){ printf("# extinction at tick %d\n",t); break; }
        }
        if(dump_every>0 && (t%dump_every==0 || t==TICKS-1)){
            CK(cudaMemcpy(h_grid,d_grid,N*sizeof(uint32_t),cudaMemcpyDeviceToHost));
            dump_pgm(h_grid,H,W,(int)seed,t);
        }
    }
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms=0; cudaEventElapsedTime(&ms,t0,t1);
    double cell_updates = (double)N*TICKS;
    printf("# elapsed %.1f ms  |  %.2f billion cell-updates/sec\n",
           ms, cell_updates/(ms*1e6));

    cudaFree(d_grid);cudaFree(d_res);cudaFree(d_res2);cudaFree(d_wants);
    cudaFree(d_alive);cudaFree(d_thrsum);cudaFree(d_present);
    free(h_grid);free(h_res);
    return 0;
}
