<div align="center">
    <h1>Cudalis</h1>
    <p>Portable Python machine learning container build and run tool.</p>
    <p><b> ⚠️ This project is under development. ⚠️ </b></p>
</div>

<p align="center">
    <img alt="" src="https://img.shields.io/badge/LICENSE-WTFPL-blueviolet?style=for-the-badge&labelColor=black&link=.%2FLICENSE">
</p>

# ⚡ Quick Start

## Requirements

- Rust

## Setup

```bash
git clone https://github.com/kerthical/cudalis.git
cd cudalis
cargo build --release
```

## Usage

This command automatically creates a container with the specified version of Python, PyTorch, and CUDA depends on
environment.

```bash
cudalis [--python/-p <VERSION>] [--torch/-t <VERSION>] [--cuda/-c <VERSION>]
```