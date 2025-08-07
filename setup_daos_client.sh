#!/bin/bash
set -e

echo "=== DAOS 클라이언트 & 빌드 환경 자동 세팅 시작 ==="

# 1. 기본 패키지 설치
echo "[1/8] 필수 패키지 설치"
sudo yum install -y epel-release
sudo yum groupinstall -y "Development Tools"
sudo yum install -y git cmake gcc gcc-c++ make glibc-devel \
    libuuid-devel openssl-devel hwloc-devel libtool autoconf automake \
    libyaml-devel libevent-devel libaio-devel python3 python3-pip ninja-build clang patchelf

# 2. DAOS 클라이언트 설치
echo "[2/8] DAOS 클라이언트 설치"
sudo yum install -y daos-client

# 3. MPICH(DAOS ROMIO) 빌드
echo "[3/8] MPICH(DAOS ROMIO) 빌드"
cd $HOME
git clone -b v3.4.3 https://github.com/pmodels/mpich mpich-3.4.3
cd mpich-3.4.3 && ./autogen.sh
./configure --prefix=$HOME/software/mpich --enable-fortran=all --enable-romio --enable-cxx --enable-g=all --enable-debuginfo --with-device=ch3:nemesis --with-file-system=ufs+daos --with-daos=$HOME/daos-install FFLAGS=-fallow-argument-mismatch
make -j$(nproc)
make install

# 4. 병렬 HDF5 빌드
echo "[4/8] 병렬 HDF5 빌드"
cd $HOME
wget https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5-1_14_3.tar.gz
tar -xvf hdf5-1_14_3.tar.gz
cd hdf5-hdf5-1_14_3
CC=$HOME/software/mpich/bin/mpicc ./configure \
    --prefix=$HOME/hdf5-install \
    --enable-parallel \
    --enable-map-api \
    --enable-ros3-vfd \
    --enable-shared
make -j$(nproc)
make install

# 5. DAOS VOL Plugin 빌드
echo "[5/8] DAOS VOL Plugin 빌드"
cd $HOME
git clone --recurse-submodules https://github.com/HDFGroup/vol-daos.git
cd vol-daos && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$HOME/daos-vol-install \
    -DHDF5_DIR=$HOME/hdf5-install \
    -DCMAKE_C_COMPILER=$HOME/software/mpich/bin/mpicc \
    -DDAOS_INCLUDE_DIR=$HOME/daos-install/include \
    -DDAOS_LIBRARY=$HOME/daos-install/lib64/libdaos.so \
    -DDAOS_UNS_LIBRARY=$HOME/daos-install/lib64/libduns.so \
    -DBUILD_TESTING=OFF
make -j$(nproc)
make install

# 6. Python 패키지 설치
echo "[6/8] Python 패키지 설치"
pip3 install --upgrade pip
pip3 install tensorflow torch h5py mpi4py

# 7. Monkey Patch 복사
echo "[7/8] Monkey Patch 복사"
mkdir -p $HOME/monkey_patch
cp -r ./monkey_patch/* $HOME/monkey_patch/
cp -r $(dirname "$0")/monkey_patch/* $HOME/monkey_patch/

# 8. 환경변수 파일 생성
echo "[8/8] 환경변수 파일 생성"
read -p "DAOS Pool UUID 입력: " pool
read -p "DAOS Container UUID 입력: " cont

cat > $HOME/daos_env.sh <<EOL
#!/bin/bash
export DAOS_DIR=\$HOME/daos-install
export MPICH_DAOS_DIR=\$HOME/software/mpich
export HDF5_DIR=\$HOME/hdf5-install

export PATH=\$MPICH_DAOS_DIR/bin:\$PATH
export LD_LIBRARY_PATH=\$HDF5_DIR/lib:\$DAOS_DIR/lib64:\$MPICH_DAOS_DIR/lib:\$HOME/daos-vol-install/lib:\$LD_LIBRARY_PATH

export HDF5_PLUGIN_PATH=\$HOME/daos-vol-install/lib
export HDF5_VOL_CONNECTOR="daos"

export PYTHONPATH=\$HOME/monkey_patch:\$HOME/daos/src/client:\$PYTHONPATH

export DAOS_POOL=$pool
export DAOS_CONT=$cont
EOL

chmod +x $HOME/daos_env.sh

# .bashrc에 자동 로드 추가
if ! grep -q "source \$HOME/daos_env.sh" $HOME/.bashrc; then
    echo "source \$HOME/daos_env.sh" >> $HOME/.bashrc
fi

echo "=== 설치 및 환경변수 세팅 완료 ==="
echo "터미널을 새로 열거나 'source ~/daos_env.sh' 실행 후 사용하세요."

