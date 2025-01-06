
all: \
	zig-vanilla \
	zig-enhanced \
	rust-vanilla \
	rust-enhanced \
	python-vanilla-cpython \
	python-enhanced-cpython \
	python-vanilla-pypy \
	go-vanilla \
	go-enhanced \
	ts-vanilla-node \
	ts-enhanced-node

python-vanilla-cpython:
	@echo ">> Python Vanilla"
	cd python/Vanilla && uv run -p 3.13 -s main.py

python-enhanced-cpython:
	@echo ">> Python Enhanced"
	cd python/Enhanced && uv run --with numpy -p 3.13 -s main.py

python-vanilla-pypy:
	@echo ">> Python Vanilla Pypy"
	cd python/Vanilla && uv run -p pypy-3.10 -s main.py

rust-vanilla:
	@echo ">> Rust Vanilla"
	cd RS/Vanilla && cargo run --release

rust-enhanced:
	@echo ">> Rust Enhanced"
	cd RS/Enhanced && cargo run --release

zig-vanilla:
	@echo ">> Zig Vanilla"
	cd Zig/Vanilla && zig run main.zig

zig-enhanced:
	@echo ">> Zig Enhanced"
	cd Zig/Enhanced && zig run src/main.zig

go-vanilla:
	@echo ">> Go Vanilla"
	cd Go/Vanilla && go run cmd/main.go

go-enhanced:
	@echo ">> Go Enhanced"
	cd Go/Enhanced && go run cmd/main.go

ts-vanilla-node:
	@echo ">> Typescript Vanilla Node"
	cd TypeScript/Vanilla && tsc && node main.js

ts-enhanced-node:
	@echo ">> Typescript Enhanced Node"
	cd TypeScript/Enhanced && tsc && node main.js