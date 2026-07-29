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
# Arch: default sm_90 (H100). Override with `make ARCH=sm_80` for A100, etc.
# (run.sh / profile.sh auto-detect the arch from nvidia-smi when ARCH is unset.)
ARCH      ?= sm_90
NVCCFLAGS ?= -O3 -arch=$(ARCH)

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
	SEEDS="$(SEEDS)" ARCH=$(ARCH) bash run.sh

# Fast smoke test: small lattice, few ticks, one seed — confirms the pod works.
quick: $(BIN)
	@echo "Smoke test: 1024x1024, 200 ticks, seed 1"
	./$(BIN) 1024 1024 200 1 20 0

analyze:
	python3 analyze.py results

# Memory-bandwidth + timeline profiling (short run). Needs ncu and/or nsys.
profile:
	ARCH=$(ARCH) bash profile.sh

# Stitch existing artifacts into one report (no runs).
report:
	python3 combine_report.py results profile_out MITO4_REPORT.md

# ONE-SHOT: full sweep, then profiling, then combined report.
# Profiling failures (e.g. no ncu perms) do not abort the report.
everything:
	H=$(H) W=$(W) TICKS=$(TICKS) LOG_EVERY=$(LOG_EVERY) DUMP=$(DUMP) \
	  SEEDS="$(SEEDS)" ARCH=$(ARCH) bash run.sh
	-ARCH=$(ARCH) bash profile.sh
	python3 combine_report.py results profile_out MITO4_REPORT.md
	@echo ""
	@echo "=============================================="
	@echo " ALL DONE -> MITO4_REPORT.md"
	@echo "=============================================="

arch:
	@echo "Detected/selected ARCH = $(ARCH)"
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

clean:
	rm -f $(BIN) mito4_prof *.o snapshot_*.pgm

distclean: clean
	rm -rf results profile_out
