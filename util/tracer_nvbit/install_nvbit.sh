export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

# NVBIT_VERSION=1.5.5
NVBIT_VERSION=1.5.3

echo "Warning: nvbit is already included in this Git repository by default."
exit 0 

rm -rf $BASH_ROOT/nvbit_release
wget https://github.com/NVlabs/NVBit/releases/download/$NVBIT_VERSION/nvbit-Linux-x86_64-$NVBIT_VERSION.tar.bz2
tar -xf nvbit-Linux-x86_64-$NVBIT_VERSION.tar.bz2 -C $BASH_ROOT
rm nvbit-Linux-x86_64-$NVBIT_VERSION.tar.bz2


