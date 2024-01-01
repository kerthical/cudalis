<div align="center">
    <h1>Cudalis</h1>
    <p>Portable Python machine learning container build and run tool.</p>
    <p><b> ⚠️ This project is under development. ⚠️ </b></p>
</div>

<p align="center">
    <img alt="" src="https://img.shields.io/badge/LICENSE-WTFPL-blueviolet?style=for-the-badge&labelColor=black&link=.%2FLICENSE">
</p>

# ⚡ Quick Start

<details>
<summary>Download binary</summary>

Choose your platform and download binary from [release page](https://github.com/kerthical/cudalis/releases/latest).
</details>

<details>
<summary>Manual build</summary>

## Requirements

- Rust

## Setup

```bash
git clone https://github.com/kerthical/cudalis.git
cd cudalis
cargo install --path . # or `cargo build --release` to build binary local directory
```

</details>

# Usage

This command automatically creates a container with the specified version of Python, PyTorch, and CUDA depends on
environment. If you don't specify the version, it will use the latest supported version automatically.

```bash
cudalis[.exe] [--python/-p <VERSION>] [--torch/-t <VERSION>] [--cuda/-c <VERSION>]
```

# Examples

Use Python 3.8.5, PyTorch 1.7.1, and CUDA 11.0.

```bash
cudalis --python 3.8.5 --torch 1.7.1 --cuda 11.0
```

Use Python 3.8.5, PyTorch 1.7.1, and Latest supported CUDA.

```bash
cudalis --python 3.8.5 --torch 1.7.1
```