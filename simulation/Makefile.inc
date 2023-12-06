# PYTHIA configuration file.
# Generated on Tue Nov 7 15:40:32 CST 2023 with the user supplied options:
# --with-root=/cvmfs/sft.cern.ch/lcg/releases/LCG_104b_ATLAS_2/ROOT/6.28.08/x86_64-centos7-gcc11-opt

# Install directory prefixes.
PREFIX=/home/abadea/pythia8307
PREFIX_BIN=$(PREFIX)/bin
PREFIX_INCLUDE=$(PREFIX)/include
PREFIX_LIB=$(PREFIX)/lib
PREFIX_SHARE=$(PREFIX)/share/Pythia8

# Compilation flags (see ./configure --help for further documentation).
CXX=/cvmfs/sft.cern.ch/lcg/releases/gcc/11.3.0-ad0f5/x86_64-centos7/bin/g++
CXX_COMMON=-O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread
CXX_SHARED=-shared
CXX_SONAME=-Wl,-soname,
LIB_SUFFIX=.so
OBJ_COMMON=

EVTGEN_USE=false
EVTGEN_CONFIG=
EVTGEN_BIN=
EVTGEN_INCLUDE=
EVTGEN_LIB=

FASTJET3_USE=false
FASTJET3_CONFIG=
FASTJET3_BIN=
FASTJET3_INCLUDE=
FASTJET3_LIB=

HEPMC2_USE=false
HEPMC2_CONFIG=
HEPMC2_BIN=
HEPMC2_INCLUDE=
HEPMC2_LIB=

HEPMC3_USE=true
HEPMC3_CONFIG=HepMC3-config
HEPMC3_BIN=/home/abadea/hepmc3.2.6/install/bin/
HEPMC3_INCLUDE=-I/home/abadea/hepmc3.2.6/install/include
HEPMC3_LIB=-L/home/abadea/hepmc3.2.6/install/lib64 -Wl,-rpath,/home/abadea/hepmc3.2.6/install/lib64 -lHepMC3

LHAPDF5_USE=false
LHAPDF5_CONFIG=
LHAPDF5_BIN=
LHAPDF5_INCLUDE=
LHAPDF5_LIB=

LHAPDF6_USE=false
LHAPDF6_CONFIG=
LHAPDF6_BIN=
LHAPDF6_INCLUDE=
LHAPDF6_LIB=

POWHEG_USE=false
POWHEG_CONFIG=
POWHEG_BIN=
POWHEG_INCLUDE=
POWHEG_LIB=

RIVET_USE=false
RIVET_CONFIG=
RIVET_BIN=
RIVET_INCLUDE=
RIVET_LIB=

ROOT_USE=true
ROOT_CONFIG=root-config
ROOT_BIN=/cvmfs/sft.cern.ch/lcg/releases/ROOT/6.28.08-2a100/x86_64-centos7-gcc11-opt/bin/
ROOT_INCLUDE=-I/cvmfs/sft.cern.ch/lcg/releases/ROOT/6.28.08-2a100/x86_64-centos7-gcc11-opt/include
ROOT_LIB=-L/cvmfs/sft.cern.ch/lcg/releases/ROOT/6.28.08-2a100/x86_64-centos7-gcc11-opt/lib -Wl,-rpath,/cvmfs/sft.cern.ch/lcg/releases/ROOT/6.28.08-2a100/x86_64-centos7-gcc11-opt/lib -lCore

GZIP_USE=false
GZIP_CONFIG=
GZIP_BIN=
GZIP_INCLUDE=
GZIP_LIB=

PYTHON_USE=false
PYTHON_CONFIG=
PYTHON_BIN=
PYTHON_INCLUDE=
PYTHON_LIB=

MG5MES_USE=false
MG5MES_CONFIG=
MG5MES_BIN=
MG5MES_INCLUDE=
MG5MES_LIB=

OPENMP_USE=false
OPENMP_CONFIG=
OPENMP_BIN=
OPENMP_INCLUDE=
OPENMP_LIB=

MPICH_USE=false
MPICH_CONFIG=
MPICH_BIN=
MPICH_INCLUDE=
MPICH_LIB=

HDF5_USE=false
HDF5_CONFIG=
HDF5_BIN=
HDF5_INCLUDE=
HDF5_LIB=

HIGHFIVE_USE=false
HIGHFIVE_CONFIG=
HIGHFIVE_BIN=
HIGHFIVE_INCLUDE=
HIGHFIVE_LIB=
