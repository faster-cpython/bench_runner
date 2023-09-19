# Provisioning self-hosted runners

This document describes how to set up machines to serve as self-hosted runners for `bench_runner`.

These instructions are only a WFM™ rough guide: This is the sort of information that gets out of date rather quickly.

While cloud VMs will *work*, we highly recommend using bare metal machines for the most stable results.

## Linux

These instructions are for Ubuntu 22.04.  If you want to benchmark on a different Linux distribution, you will need to adapt these instructions accordingly.

### Install requirements

```bash session
sudo apt install python3 build-essential ccache gdb lcov pkg-config \
      libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
      libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
      lzma lzma-dev tk-dev uuid-dev zlib1g-dev linux-perf
```

### Enable ccache

```bash session
sudo /usr/sbin/update-ccache-symlinks
echo 'export PATH="/usr/lib/ccache:$PATH"' | tee -a ~/.bashrc
```

### Setup passwordless sudo

[Running sudo with no password](https://askubuntu.com/questions/192050/how-to-run-sudo-command-with-no-password)

## macOS

### Install Apple Developer Tools

You can simply install XCode from the store, or if you don't have graphical access, try installing "Command Line Tools for XCode" using [these instructions](https://apple.stackexchange.com/questions/107307/how-can-i-install-the-command-line-tools-completely-from-the-command-line).

### Install Homebrew

This is the easiest way to get openssl, which is required for pip downloading files from PyPI.

#### Install brew

```bash session
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Add the following to the end of `~/.zprofile`:

```bash session
eval "$(/opt/homebrew/bin/brew shellenv)"
```

#### Install brew packages

```bash session
brew install openssl jq python@3.9 coreutils ccache
```

## Microsoft Windows

### Adjust settings and permissions

In Settings, turn on "Developer Mode" (which enables symlinks)

In Settings, turn on "Change execution policy to allow running local PowerShell scripts without signing"

In an administrator PowerShell terminal, run `Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process`

### Install Visual Studio Community Edition 2022

Install Visual Studio Community Edition 2022 from the Microsoft Store.

Install the following components:

- MSVC v143 VS 2022 x64/x86 build tools
- Windows SDK version 10.0.22621.0
- Windows SDK version 10.0.19041.0

### Install git for Windows

Install [git for Windows](https://git-scm.com/download/win).

### Install Python

Install Python 3.9 or later from [python.org](https://python.org), and install for all users.
