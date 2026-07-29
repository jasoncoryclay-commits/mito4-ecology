# MITO-4 Ecology — Makefile
# Targets:  make            (build)
#           make everything (sweep + profile + single combined MITO4_REPORT.md)  <-- one-shot
#           make run        (build + full 5-seed sweep + analyze via run.sh)
#           make quick      (single short seed-1 run to smoke-test the pod)
#           make analyze    (re-score existing results/)
#           make profile    (ncu/nsys: achieved DRAM bandwidth + timeline)
#           make report     (stitch existing results+profile into MITO4_REPORT.md)
#           make arch       (print the detected GPU arch)
#           make clean      (remove binary + build artifacts)
#           make distclean  (also remove results/)

NVCC      ?= nvcc
# Arch: leave EMPTY by default so run.sh/profile.sh auto-detect from nvidia-smi
# (H100->sm_90, A100->sm_80, L40/4090->sm_89, V100->sm_70).
# Override explicitly if needed:  make ARCH=sm_80
ARCH      ?=
ARCH_FLAG := $(if $(ARCH),-arch=$(ARCH),-arch=sm_80)
NVCCFLAGS ?= -O3 $(ARCH_FLAG)

# Run parameters (override on the command line: make run H=16384 W=16384)
H         ?= 8192
W         ?= 8192
TICKS     ?= 5000
LOG_EVERY ?= 100
DUMP      ?= 0
SEEDS     ?= 1 2 3 4 5

BIN = mito4
SRC = mito4_kernel.cu

.PHONY: all everything run quick analyze profile report arch clean distclean

all: $(BIN)

$(BIN): $(SRC)
	@echo "Building $(BIN) with $(NVCC) $(NVCCFLAGS)"
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(BIN)

# Full experiment via run.sh (handles logging + analysis + GPU util capture)
run:
	H=$(H) W=$(W) TICKS=$(TICKS) LOG_EVERY=$(LOG_EVERY) DUMP=$(DUMP) \
	SEEDS="$(SEEDS)" ARCH="$(ARCH)" bash run.sh

# Fast smoke test: small lattice, few ticks, one seed — confirms the pod works.
quick: $(BIN)
	@echo "Smoke test: 1024x1024, 200 ticks, seed 1"
	./$(BIN) 1024 1024 200 1 20 0

analyze:
	python3 analyze.py results

# Memory-bandwidth + timeline profiling (short run). Needs ncu and/or nsys.
profile:
	ARCH="$(ARCH)" bash profile.sh

# Stitch existing artifacts into one report (no runs).
report:
	python3 combine_report.py results profile_out MITO4_REPORT.md

# ONE-SHOT: full sweep, then profiling, then combined report.
# Profiling failures (e.g. no ncu perms) do not abort the report.
everything:
	H=$(H) W=$(W) TICKS=$(TICKS) LOG_EVERY=$(LOG_EVERY) DUMP=$(DUMP) \
	  SEEDS="$(SEEDS)" ARCH="$(ARCH)" bash run.sh
	-ARCH="$(ARCH)" bash profile.sh
	python3 combine_report.py results profile_out MITO4_REPORT.md
	@echo ""
	@echo "=============================================="
	@echo " ALL DONE -> MITO4_REPORT.md"
	@echo "=============================================="

arch:
	@echo "Makefile ARCH override = '$(ARCH)' (empty => scripts auto-detect)"
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
	@name=$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); \
	 case "$$name" in \
	   *H100*|*H200*) echo "auto-detect -> sm_90" ;; \
	   *A100*) echo "auto-detect -> sm_80" ;; \
	   *L40*|*4090*|*L4*) echo "auto-detect -> sm_89" ;; \
	   *V100*) echo "auto-detect -> sm_70" ;; \
	   *) echo "auto-detect -> sm_90 (default)" ;; \
	 esac

clean:
	rm -f $(BIN) mito4_prof *.o snapshot_*.pgm

distclean: clean
	rm -rf results profile_out
