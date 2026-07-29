# MITO-4 Ecology — Makefile
# Targets:  make            (build)
#           make run        (build + full 5-seed sweep + analyze via run.sh)
#           make quick      (single short seed-1 run to smoke-test the pod)
#           make analyze    (re-score existing results/)
#           make arch       (print the detected GPU arch)
#           make clean      (remove binary + build artifacts)
#           make distclean  (also remove results/)

NVCC      ?= nvcc
# Auto-detect arch from the GPU name if ARCH not supplied.
ARCH      ?= $(shell \
	name=$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); \
	case "$$name" in \
	  *H100*|*H200*) echo sm_90 ;; \
	  *A100*)        echo sm_80 ;; \
	  *L40*|*4090*|*L4*) echo sm_89 ;; \
	  *V100*)        echo sm_70 ;; \
	  *)             echo sm_90 ;; \
	esac)
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

.PHONY: all run quick analyze arch clean distclean

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

arch:
	@echo "Detected/selected ARCH = $(ARCH)"
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"

clean:
	rm -f $(BIN) *.o snapshot_*.pgm

distclean: clean
	rm -rf results
