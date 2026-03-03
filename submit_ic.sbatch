#!/bin/bash 
#SBATCH --job-name=idm_n2_1e-2GeV_envelope_ic
#SBATCH --output=music_run_%j.out
#SBATCH --error=music_run_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arifchu@usc.edu

module purge
# Load required modules
module load gcc/13.2.0-gcc-13.2.0-w55nxkl
module load gsl/2.7.1-gcc-11.3.1-jex6yq2
module load mpich
module load hdf5/1.8.23-gcc-11.3.1-2ckvz5j

# Set environment
export OMP_NUM_THREADS=16
export LD_LIBRARY_PATH=$(dirname $(find $EBROOTGSL -name libgsl.so.19)):$LD_LIBRARY_PATH

cd "/resnick/groups/carnegie_poc/achu/n2_1e-2GeV_envelope_fixed"

# Clean ic directory
echo "Cleaning ic directory..."
rm -rf ic
mkdir -p ic

# Check files
echo "Checking configuration..."
echo "Working directory: $(pwd)"
echo "MUSIC executable: $(ls -la MUSIC 2>/dev/null || echo 'NOT FOUND')"
echo "music.conf: $(ls -la music.conf 2>/dev/null || echo 'NOT FOUND')"

echo "Starting MUSIC at: $(date)"
echo "Running: ./MUSIC music.conf"
echo ""

./MUSIC music.conf
EXIT_CODE=$?

echo ""
echo "MUSIC finished at: $(date)"
echo "Exit code: $EXIT_CODE"

# Check final results of MUSIC
if [ $EXIT_CODE -eq 0 ] && [ -f "ic/ic_gadget" ]; then
    SIZE=$(stat -c%s "ic/ic_gadget" 2>/dev/null || stat -f%z "ic/ic_gadget" 2>/dev/null)
    echo "SUCCESS: ic/ic_gadget created ($SIZE bytes)"

    # Quick validation
    if [ $SIZE -gt 256 ]; then
        echo "File size appears valid"
    else
        echo "WARNING: File may be too small ($SIZE bytes)"
    fi
else
    echo "ERROR: MUSIC failed or ic_gadget not created"
    echo "Check error log for details"
fi
